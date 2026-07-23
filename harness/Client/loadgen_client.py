# ============================================================================
# loadgen_client.py
# -----------------
# MLPerf LoadGen client implementation
# Supports both Offline and Server scenarios
# ============================================================================

import os
import sys
import time
import logging
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import array
import base64
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from io import BytesIO

# Try to import pandas for dataset operations
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add parent directories to path for imports
harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harness_root)

# Import base client and dataset processor
try:
    from Client.base_client import BaseClient
    from data.dataset_processor import DatasetProcessor
    # Try to import config availability flag
    try:
        from data.dataset_processor import CONFIG_AVAILABLE
    except ImportError:
        CONFIG_AVAILABLE = False
except ImportError:
    # Try relative imports if absolute fails
    from .base_client import BaseClient
    import sys
    import os
    sys.path.insert(0, os.path.dirname(harness_root))
    from harness.data.dataset_processor import DatasetProcessor
    try:
        from harness.data.dataset_processor import CONFIG_AVAILABLE
    except ImportError:
        CONFIG_AVAILABLE = False

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    sys.exit(1)

# Import tokenizer for API mode
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class LoadGenClient(BaseClient):
    """
    MLPerf LoadGen client implementation.
    
    Base class for LoadGen clients (Offline and Server scenarios).
    """
    
    def __init__(self,
                 model_name: str,
                 dataset_path: str,
                 scenario: str = "Offline",
                 test_mode: str = "performance",
                 api_server_url: Optional[str] = None,
                 api_server_urls: Optional[List[str]] = None,
                 batch_size: int = 13368,
                 num_samples: int = 13368,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize LoadGen client.
        
        Args:
            model_name: Model name or path
            dataset_path: Path to dataset file
            scenario: LoadGen scenario ("Offline" or "Server")
            test_mode: Test mode ("performance" or "accuracy")
            api_server_url: Optional API server URL (if using remote server) - backward compatible
            api_server_urls: Optional list of API server URLs for load balancing
            batch_size: Batch size for processing
            num_samples: Number of samples for testing
            config: Additional configuration
        """
        super().__init__("loadgen", model_name, dataset_path, config)
        
        self.scenario = scenario
        self.test_mode = test_mode
        
        # Handle load balancing: prefer api_server_urls, fall back to api_server_url
        if api_server_urls:
            self.api_server_urls = [url.rstrip('/') for url in api_server_urls]
            self.api_server_url = self.api_server_urls[0]  # Primary URL for backward compatibility
            self.load_balancing = True
            self.load_balance_strategy = config.get('load_balance_strategy', 'round_robin') if config else 'round_robin'
            self.current_server_index = 0
            self.failed_servers = set()  # Track servers that have failed
            self.max_retries_per_server = config.get('max_retries_per_server', 3) if config else 3
            self.logger.info(f"Load balancing enabled with {len(self.api_server_urls)} servers: {self.api_server_urls}")
            self.logger.info(f"Load balance strategy: {self.load_balance_strategy}")
        else:
            self.api_server_urls = None
            self.api_server_url = api_server_url.rstrip('/') if api_server_url else None
            self.load_balancing = False
            self.current_server_index = 0
            self.failed_servers = set()
        
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        # HTTP session with connection pooling — reuses TCP connections
        # instead of opening a new one per request (avoids overwhelming proxy)
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=128,    # number of connection pools (per host)
            pool_maxsize=8192,       # max connections per pool
            max_retries=0,           # we handle retries ourselves
            pool_block=False,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Dataset processor
        self.dataset: Optional[DatasetProcessor] = None

        # API mode components
        self.tokenizer = None
        self.completions_endpoint = None
        self.chat_completions_endpoint = None
        self.health_endpoint = None
        self.server_ready = False
        
        # Endpoint type: 'completions' or 'chat_completions'
        self.endpoint_type = config.get('endpoint_type', 'completions') if config else 'completions'
        if self.endpoint_type not in ['completions', 'chat_completions']:
            raise ValueError(f"Invalid endpoint_type: {self.endpoint_type}. Must be 'completions' or 'chat_completions'")
        
        # Max tokens configuration - will be updated after dataset is loaded if dataset config has it
        self.max_tokens = self._determine_max_tokens(model_name, config, test_mode)
        self.logger.info(f"Initial max_tokens: {self.max_tokens} for model: {model_name} (test_mode: {test_mode})")
        
        # Sampling parameters - can be different for accuracy vs performance
        # For gpt-oss-120b, use same parameters for both modes: temperature=1.0, top_k=-1, top_p=1.0
        model_lower = model_name.lower()
        is_gpt_oss = 'gpt-oss' in model_lower or 'gpt_oss' in model_lower or 'gptoss' in model_lower
        
        if is_gpt_oss:
            # gpt-oss-120b uses same sampling params for both perf and accuracy
            self.temperature = config.get('temperature', 1.0) if config else 1.0
            self.top_k = config.get('top_k', -1) if config else -1
            self.top_p = config.get('top_p', 1.0) if config else 1.0
            # Set accuracy params to same values
            self.accuracy_temperature = config.get('accuracy_temperature', 1.0) if config else 1.0
            self.accuracy_top_k = config.get('accuracy_top_k', -1) if config else -1
            self.accuracy_top_p = config.get('accuracy_top_p', 1.0) if config else 1.0
            self.logger.debug(f"GPT-OSS model detected - Sampling parameters initialized:")
            self.logger.debug(f"  Performance mode: temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}")
            self.logger.debug(f"  Accuracy mode: temperature={self.accuracy_temperature}, top_k={self.accuracy_top_k}, top_p={self.accuracy_top_p}")
        else:
            # Default behavior for other models
            self.temperature = config.get('temperature', 0.0) if config else 0.0
            self.top_k = config.get('top_k', 1) if config else 1
            self.top_p = config.get('top_p', 1.0) if config else 1.0
            # Accuracy mode parameters (if specified, override for accuracy mode)
            self.accuracy_temperature = config.get('accuracy_temperature', None) if config else None
            self.accuracy_top_k = config.get('accuracy_top_k', None) if config else None
            self.accuracy_top_p = config.get('accuracy_top_p', None) if config else None
            self.logger.debug(f"Sampling parameters initialized:")
            self.logger.debug(f"  Performance mode: temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}")
            if self.accuracy_temperature is not None or self.accuracy_top_k is not None or self.accuracy_top_p is not None:
                self.logger.debug(f"  Accuracy mode overrides: temperature={self.accuracy_temperature}, top_k={self.accuracy_top_k}, top_p={self.accuracy_top_p}")
            else:
                self.logger.debug(f"  Accuracy mode: using performance mode parameters")
        
        # SGLang-specific: use input_ids directly instead of text
        # Auto-detect SGLang backend from server_config
        backend = None
        if config and 'server_config' in config:
            backend = config['server_config'].get('backend', 'vllm')
        
        # Set use_input_ids if backend is SGLang or explicitly set in config
        self.use_input_ids = config.get('use_input_ids', False) if config else False
        if backend and backend.lower() == 'sglang' and not self.use_input_ids:
            # Auto-enable input_ids mode for SGLang backend
            self.use_input_ids = True
            self.logger.info("Auto-detected SGLang backend, enabling input_ids mode")
        
        self.sglang_endpoint = config.get('sglang_endpoint', '/generate') if config else '/generate'
        
        self.use_guided_decoding = config.get('use_guided_decoding', False) if config else False
        
        # Offline scenario: send requests back-to-back instead of batching
        self.offline_back_to_back = config.get('offline_back_to_back', False) if config else False
        
        # For async offline_back_to_back: max concurrent requests (default: 10)
        # This controls how many requests are in-flight at once
        self.offline_async_concurrency = config.get('offline_async_concurrency', 10) if config else 10
        
        # Log offline_back_to_back status for debugging
        if self.scenario == "Offline":
            if self.offline_back_to_back:
                self.logger.info("=" * 80)
                self.logger.info("CLIENT: offline_back_to_back = True (ASYNC MODE)")
                self.logger.info("CLIENT: Will send requests individually (one per API call)")
                self.logger.info("CLIENT: Requests will be sent asynchronously (fire-and-forget)")
                self.logger.info(f"CLIENT: Max concurrent requests: {self.offline_async_concurrency}")
                self.logger.info("CLIENT: Responses will be processed as they arrive")
                self.logger.info("=" * 80)
            else:
                self.logger.info("=" * 80)
                self.logger.info("CLIENT: offline_back_to_back = False (default)")
                self.logger.info(f"CLIENT: Will batch requests with batch_size={self.batch_size}")
                self.logger.info("=" * 80)
        
        # Debug mode for accuracy mode
        self.debug_mode = config.get('debug_mode', False) if config else False
        
        # Token statistics tracking for histograms and summary
        # Enable if either debug_mode or print_token_stats is True
        self.print_token_stats = config.get('print_token_stats', False) if config else False
        self.input_token_counts = []
        self.output_token_counts = []
        self.token_ratios = []

        # CRITICAL FIX: Storage for response arrays to prevent garbage collection
        # LoadGen stores pointers to these arrays and reads them later when writing accuracy JSON
        self.response_arrays = {}  # {query_id: numpy_array or array.array}

        # Accuracy results storage (matching offline_sut.py format)
        # Store results in format: {query_id: {"output_ids": [], "output_text": "", "metadata": {}}}
        self.results = {} if self.test_mode == "accuracy" else None
        
        # Server scenario specific components (for async processing)
        self.num_workers = config.get('num_workers', 1) if config else 1
        self.worker_threads: List[Optional[threading.Thread]] = []
        self.query_queue: Optional[queue.Queue] = None
        self.workers_started = False
        
        # Initialize endpoints (for load balancing, use primary URL for endpoints)
        if self.load_balancing:
            # For load balancing, endpoints are constructed per-request
            primary_url = self.api_server_urls[0]
            self.completions_endpoint = f"{primary_url}/v1/completions"
            self.chat_completions_endpoint = f"{primary_url}/v1/chat/completions"
            self.health_endpoint = f"{primary_url}/health"
        elif self.api_server_url:
            self.api_server_url = self.api_server_url.rstrip('/')
            self.completions_endpoint = f"{self.api_server_url}/v1/completions"
            self.chat_completions_endpoint = f"{self.api_server_url}/v1/chat/completions"
            self.health_endpoint = f"{self.api_server_url}/health"
            
            # Validate endpoint based on backend config
            self._validate_endpoint()
    
    def initialize(self) -> None:
        """Initialize LoadGen client."""
        self.logger.info(f"Initializing LoadGen client (scenario: {self.scenario})")
        
        # Load dataset with configuration support
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        
        # Extract dataset configuration from config if available
        dataset_name = self.config.get('dataset_name')
        input_column = self.config.get('input_column')
        input_ids_column = self.config.get('input_ids_column')
        output_column = self.config.get('output_column')
        config_dir = self.config.get('config_dir')
        
        # Determine total_sample_count to pass to DatasetProcessor
        # If dataset config exists and has total_sample_count, use None to let config handle it
        # Otherwise, use self.num_samples
        # We'll check for config first by trying to load it
        total_sample_count_for_loader = self.num_samples
        try:
            if CONFIG_AVAILABLE and dataset_name:
                from data.dataset_config import DatasetConfigLoader
                config_loader = DatasetConfigLoader(config_dir=config_dir)
                dataset_config = config_loader.load_dataset_config(dataset_name, self.model_name)
                if dataset_config and dataset_config.total_sample_count is not None:
                    # Config has total_sample_count - pass None to let DatasetProcessor use config value
                    # This ensures we load all samples from file, then limit based on config
                    total_sample_count_for_loader = None
                    self.logger.info(f"Dataset config specifies total_sample_count={dataset_config.total_sample_count}, will load all samples and use config value")
        except Exception as e:
            # If config loading fails, fall back to using self.num_samples
            self.logger.debug(f"Could not pre-check dataset config: {e}, using num_samples={self.num_samples}")
        
        self.dataset = DatasetProcessor(
            dataset_path=self.dataset_path,
            model_name=self.model_name,
            total_sample_count=total_sample_count_for_loader,
            dataset_name=dataset_name,
            input_column=input_column,
            input_ids_column=input_ids_column,
            output_column=output_column,
            config_dir=config_dir
        )
        
        # Update max_tokens from dataset config if available
        # Note: For gpt-oss-120b, max_tokens may be test_mode-dependent
        if hasattr(self.dataset, 'dataset_config') and self.dataset.dataset_config:
            dataset_max_tokens = self.dataset.dataset_config.model_specific.get('max_tokens')
            if dataset_max_tokens is not None:
                # Check if this is a test_mode-specific value or should override
                # For gpt-oss-120b, dataset configs have different max_tokens for perf vs accuracy
                # The dataset name itself indicates which mode (perf_eval_ref vs acc_eval_ref)
                self.max_tokens = int(dataset_max_tokens)
                self.logger.info(f"Updated max_tokens from dataset config: {self.max_tokens}")
        
        # Update num_samples from dataset config if available
        # This ensures we use the correct sample count from config (e.g., 6396 for perf_eval_ref, 4395 for acc_eval_ref)
        # The dataset.total_sample_count may have been set from config, so we should use it
        if hasattr(self.dataset, 'dataset_config') and self.dataset.dataset_config:
            config_total = self.dataset.dataset_config.total_sample_count
            if config_total is not None:
                # Dataset config specified a total_sample_count - use it
                # Limit to actual dataset size
                actual_dataset_size = len(self.dataset.input_ids)
                new_num_samples = min(config_total, actual_dataset_size)
                if new_num_samples != self.num_samples:
                    old_num_samples = self.num_samples
                    self.num_samples = new_num_samples
                    self.logger.info(f"Updated num_samples from dataset config: {old_num_samples} -> {self.num_samples} (config: {config_total}, actual: {actual_dataset_size})")
                else:
                    self.logger.info(f"Using num_samples: {self.num_samples} (matches dataset config: {config_total})")
        elif self.dataset.total_sample_count is not None and self.dataset.total_sample_count != self.num_samples:
            # Fallback: use dataset's total_sample_count if it differs from our num_samples
            # This handles cases where dataset was limited during loading
            actual_dataset_size = len(self.dataset.input_ids)
            new_num_samples = min(self.dataset.total_sample_count, actual_dataset_size)
            if new_num_samples != self.num_samples:
                old_num_samples = self.num_samples
                self.num_samples = new_num_samples
                self.logger.info(f"Updated num_samples to match dataset: {old_num_samples} -> {self.num_samples} (dataset total: {self.dataset.total_sample_count}, actual: {actual_dataset_size})")
        
        # Print dataset statistics
        stats = self.dataset.get_statistics()
        self.logger.info("=" * 60)
        self.logger.info("Dataset Statistics")
        self.logger.info("=" * 60)
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        # Print final sampling parameters and max_tokens summary
        self._print_sampling_summary()
        
        self.logger.info("=" * 60)
        
        # Initialize tokenizer if using API mode (needed for vLLM detokenization)
        if self.api_server_url:
            self._initialize_tokenizer()
            
            # For vLLM: if we have input_ids but no text, detokenize them
            # This must happen before waiting for server, as we need text for vLLM API calls
            # Exception: For gpt-oss-120b, check if text_input column exists (it should for vLLM)
            if not self.use_input_ids and self.tokenizer and self.dataset:
                # Check if dataset has text_input column (for gpt-oss-120b)
                has_text_input = False
                if hasattr(self.dataset, 'processed_data') and PANDAS_AVAILABLE:
                    df = self.dataset.processed_data
                    if 'text_input' in df.columns and len(self.dataset.input) == 0:
                        # Extract text_input column if it exists
                        self.dataset.input = df['text_input'].tolist()
                        has_text_input = len(self.dataset.input) > 0
                        if has_text_input:
                            self.logger.info(f"Using 'text_input' column directly for vLLM (no detokenization needed)")
                
                # Only detokenize if we still don't have text
                if len(self.dataset.input_ids) > 0 and len(self.dataset.input) == 0 and not has_text_input:
                    self.logger.info("Detokenizing input_ids to text for vLLM (dataset has no text field)")
                    self._detokenize_dataset()
            
            self._wait_for_server_ready()
        
        # Initialize server scenario components if needed
        if self.scenario == "Server" and self.api_server_url:
            self._initialize_server_components()
            # Note: Workers will be started when first query arrives or can be started explicitly
        
        self.is_initialized = True
        self.logger.info("LoadGen client initialized successfully")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the LoadGen client.
        
        Note: For LoadGen clients, the actual execution is handled by
        the harness calling issue_query() and flush_queries().
        This method marks the client as running.
        
        Returns:
            Dictionary with client status
        """
        self.is_running = True
        self.logger.info("LoadGen client is running")
        return {
            'status': 'running',
            'scenario': self.scenario,
            'test_mode': self.test_mode
        }
    
    def _initialize_server_components(self):
        """Initialize components for server scenario with async processing."""
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.workers_started = False
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for API mode."""
        if not TOKENIZER_AVAILABLE:
            self.logger.warning("Transformers not available, tokenizer not initialized")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.logger.info("Tokenizer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _detokenize_dataset(self):
        """Detokenize input_ids to text for vLLM when dataset has no text field."""
        if not self.tokenizer or not self.dataset:
            return
        
        if len(self.dataset.input_ids) == 0:
            return
        
        if len(self.dataset.input) > 0:
            # Already has text, no need to detokenize
            return
        
        self.logger.info(f"Detokenizing {len(self.dataset.input_ids)} samples...")
        self.dataset.input = []
        
        for i, input_ids in enumerate(self.dataset.input_ids):
            try:
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                self.dataset.input.append(text)
            except Exception as e:
                self.logger.warning(f"Error detokenizing sample {i}: {e}")
                # Fallback: convert to string representation
                self.dataset.input.append(" ".join([str(t) for t in input_ids]))
        
        self.logger.info(f"Successfully detokenized {len(self.dataset.input)} samples")
    
    def _determine_max_tokens(self, model_name: str, config: Optional[Dict[str, Any]], test_mode: str = "performance") -> int:
        """
        Determine max_tokens based on config, server_config, dataset_config, or model name.
        
        Priority order:
        1. config['max_tokens'] (explicit client config)
        2. config['server_config']['config']['max_tokens'] (server config)
        3. config['server_config']['config']['api_server_args'] with --max-model-len or --max-num-seqs
        4. dataset_config.model_specific.get('max_tokens')
        5. Model name-based defaults (test_mode-aware for gpt-oss-120b)
        6. Default: 1024
        
        Defaults:
        - deepseek-r1: 20000
        - llama3.1-8b: 128
        - llama2-70b: 1024
        - gpt-oss-120b: 10240 (performance), 32768 (accuracy)
        - default: 1024
        """
        # Priority 1: Check if explicitly set in config
        if config and 'max_tokens' in config:
            return int(config['max_tokens'])
        
        # Priority 2: Check server_config
        if config and 'server_config' in config:
            server_config = config['server_config']
            # Check in server config dict directly
            if 'max_tokens' in server_config:
                return int(server_config['max_tokens'])
            # Check in server config['config'] dict
            if 'config' in server_config and isinstance(server_config['config'], dict):
                server_config_dict = server_config['config']
                if 'max_tokens' in server_config_dict:
                    return int(server_config_dict['max_tokens'])
                # Check for max_new_tokens (alternative name)
                if 'max_new_tokens' in server_config_dict:
                    return int(server_config_dict['max_new_tokens'])
                # Check api_server_args for --max-model-len or --max-num-seqs
                if 'api_server_args' in server_config_dict:
                    args = server_config_dict['api_server_args']
                    if isinstance(args, list):
                        for i, arg in enumerate(args):
                            if arg in ['--max-model-len', '--max-num-seqs'] and i + 1 < len(args):
                                try:
                                    return int(args[i + 1])
                                except (ValueError, IndexError):
                                    pass
        
        # Priority 3: Check dataset_config (will be available after initialize)
        # This is checked later in initialize() method after dataset is loaded
        
        # Priority 4: Determine from model name (test_mode-aware for gpt-oss-120b)
        model_lower = model_name.lower()
        if 'deepseek' in model_lower and 'r1' in model_lower:
            return 20000
        elif 'gpt-oss' in model_lower or 'gpt_oss' in model_lower or 'gptoss' in model_lower:
            if '120b' in model_lower or '120-b' in model_lower:
                # gpt-oss-120b has different max_tokens for perf vs accuracy
                if test_mode == "accuracy":
                    return 32768
                else:  # performance
                    return 10240
        elif 'llama3.1' in model_lower or 'llama-3.1' in model_lower or 'llama3_1' in model_lower:
            if '8b' in model_lower or '8-b' in model_lower:
                return 128
        elif 'llama2' in model_lower or 'llama-2' in model_lower:
            if '70b' in model_lower or '70-b' in model_lower:
                return 1024
        
        # Default
        return 1024
    
    def _validate_endpoint(self):
        """Validate that the requested endpoint exists for the backend."""
        if not self.api_server_url:
            return
        
        backend = self.config.get('backend', 'vllm') if self.config else 'vllm'
        
        # Load backend config to check available endpoints
        try:
            from data.backend_config import BackendConfigLoader
            backend_loader = BackendConfigLoader()
            backend_config = backend_loader.load_backend_config(backend)
            
            # Check if endpoint is available
            available_endpoints = backend_config.get('endpoints', [])
            if self.endpoint_type not in available_endpoints:
                raise ValueError(
                    f"Endpoint '{self.endpoint_type}' is not available for backend '{backend}'. "
                    f"Available endpoints: {available_endpoints}"
                )
            
            self.logger.info(f"Validated endpoint '{self.endpoint_type}' for backend '{backend}'")
        except ImportError:
            # Backend config not available, skip validation
            self.logger.warning("Backend config loader not available, skipping endpoint validation")
        except Exception as e:
            # If backend config doesn't exist or other error, log warning but continue
            self.logger.warning(f"Could not validate endpoint: {e}")
    
    def _wait_for_server_ready(self, timeout: int = 600):
        """Wait for API server(s) to become ready."""
        if self.load_balancing:
            # Wait for at least one server to be ready
            self.logger.info(f"Waiting for at least one API server to be ready from {len(self.api_server_urls)} servers (timeout: {timeout}s)")
            ready_servers = []
            
            for url in self.api_server_urls:
                health_endpoint = f"{url}/health"
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        response = self._session.get(health_endpoint, timeout=10)
                        if response.status_code == 200:
                            self.logger.info(f"API server at {url} is ready!")
                            ready_servers.append(url)
                            break
                    except Exception as e:
                        self.logger.debug(f"API server at {url} not ready: {e}")
                    
                    time.sleep(2)
            
            if ready_servers:
                self.server_ready = True
                self.logger.info(f"{len(ready_servers)}/{len(self.api_server_urls)} servers are ready")
                # Remove failed servers from the list
                self.api_server_urls = [url for url in self.api_server_urls if url in ready_servers]
                if len(self.api_server_urls) != len(ready_servers):
                    self.logger.warning(f"Some servers are not ready. Using {len(self.api_server_urls)} available servers")
            else:
                raise RuntimeError(f"No API servers became ready within {timeout} seconds")
        elif self.api_server_url:
            self.logger.info(f"Waiting for API server at {self.api_server_url} (timeout: {timeout}s)")
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(self.health_endpoint, timeout=10)
                    if response.status_code == 200:
                        self.logger.info("API server is ready!")
                        self.server_ready = True
                        return
                except Exception as e:
                    self.logger.debug(f"API server not ready: {e}")
                
                time.sleep(2)
            
            raise RuntimeError(f"API server at {self.api_server_url} did not become ready within {timeout} seconds")
    
    def _get_next_server_url(self) -> str:
        """Get the next server URL using load balancing strategy."""
        if not self.load_balancing or not self.api_server_urls:
            return self.api_server_url
        
        # Filter out failed servers (if any are marked as permanently failed)
        available_servers = [url for url in self.api_server_urls if url not in self.failed_servers]
        if not available_servers:
            # If all servers failed, reset and try all again
            self.logger.warning("All servers marked as failed, resetting and trying all servers")
            self.failed_servers.clear()
            available_servers = self.api_server_urls
        
        if self.load_balance_strategy == 'round_robin':
            # Round-robin: cycle through servers
            url = available_servers[self.current_server_index % len(available_servers)]
            self.current_server_index = (self.current_server_index + 1) % len(available_servers)
            return url
        elif self.load_balance_strategy == 'random':
            # Random: pick a random server
            return random.choice(available_servers)
        else:
            # Default to round-robin
            url = available_servers[self.current_server_index % len(available_servers)]
            self.current_server_index = (self.current_server_index + 1) % len(available_servers)
            return url
    
    def _get_endpoints_for_url(self, base_url: str) -> Dict[str, str]:
        """Get endpoint URLs for a given base URL."""
        return {
            'completions': f"{base_url}/v1/completions",
            'chat_completions': f"{base_url}/v1/chat/completions",
            'health': f"{base_url}/health",
            'sglang': f"{base_url}{self.sglang_endpoint}"
        }
    
    def _send_request_with_retry(self, endpoint: str, payload: Dict[str, Any], server_url: str, max_retries: int = 3) -> requests.Response:
        """Send API request with retry logic and load balancing fallback."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self._session.post(endpoint, json=payload, timeout=None)
                if response.status_code == 200:
                    return response
                else:
                    # Non-200 status code - might be server error
                    last_exception = RuntimeError(f"API request failed: {response.status_code} - {response.text}")
                    if self.load_balancing and attempt < max_retries - 1:
                        # Try next server
                        self.logger.warning(f"Server {server_url} returned status {response.status_code}, trying next server...")
                        server_url = self._get_next_server_url()
                        endpoints = self._get_endpoints_for_url(server_url)
                        # Update endpoint based on original endpoint type
                        if '/chat/completions' in endpoint:
                            endpoint = endpoints['chat_completions']
                        elif '/completions' in endpoint:
                            endpoint = endpoints['completions']
                        elif '/generate' in endpoint:
                            endpoint = endpoints['sglang']
                        continue
            except (requests.exceptions.RequestException, ConnectionError) as e:
                last_exception = e
                if self.load_balancing and attempt < max_retries - 1:
                    # Try next server on connection error
                    self.logger.warning(f"Connection error to {server_url}: {e}, trying next server...")
                    server_url = self._get_next_server_url()
                    endpoints = self._get_endpoints_for_url(server_url)
                    # Update endpoint based on original endpoint type
                    if '/chat/completions' in endpoint:
                        endpoint = endpoints['chat_completions']
                    elif '/completions' in endpoint:
                        endpoint = endpoints['completions']
                    elif '/generate' in endpoint:
                        endpoint = endpoints['sglang']
                    continue
                else:
                    # No more retries or not load balancing
                    break
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"API request failed after {max_retries} attempts")
    
    @abstractmethod
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """
        Process query samples from MLPerf Loadgen.
        
        This method must be implemented by subclasses (Offline/Server).
        
        Args:
            query_samples: List of MLPerf QuerySample objects
        """
        pass
    
    @abstractmethod
    def flush_queries(self) -> None:
        """
        Flush any pending queries.
        MLPerf Loadgen callback.
        """
        pass
    
    def get_sut(self):
        """
        Get SystemUnderTest object for LoadGen.
        
        Returns:
            LoadGen SUT object constructed from issue_query and flush_queries methods
        """
        return lg.ConstructSUT(self.issue_query, self.flush_queries)
    
    def get_qsl(self):
        """
        Get QuerySampleLibrary object for LoadGen.
        
        Returns:
            LoadGen QSL object constructed from dataset and callbacks
        """
        if not self.dataset:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")
        
        # Use whatever dataset size is available (whatever loadgen sends down)
        # Use the actual number of samples loaded in the dataset
        total_count = len(self.dataset.input_ids)
        
        # performance_count should be the number of samples to use for testing
        # This is self.num_samples (which may have been updated from config)
        performance_count = min(self.num_samples, total_count)
        
        self.logger.info(f"Constructing QSL: total_count={total_count}, performance_count={performance_count}, num_samples={self.num_samples}")
        
        return lg.ConstructQSL(
            total_count,
            performance_count,
            self._load_samples_to_ram,
            self._unload_samples_from_ram
        )
    
    def _load_samples_to_ram(self, query_sample_indices):
        """
        LoadGen callback: Load samples to RAM.
        
        Args:
            query_sample_indices: List of QuerySampleIndex objects
        """
        # Samples are already loaded in DatasetProcessor
        # This is a no-op as samples are pre-loaded
        pass
    
    def _unload_samples_from_ram(self, query_sample_indices):
        """
        LoadGen callback: Unload samples from RAM.
        
        Args:
            query_sample_indices: List of QuerySampleIndex objects
        """
        # Samples are kept in memory for the duration of the test
        # This is a no-op
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        # Print token statistics if enabled (before cleanup)
        # Note: This is a fallback - main printing happens in base_harness after test completes
        if (hasattr(self, 'print_token_stats') and self.print_token_stats) or \
           (hasattr(self, 'debug_mode') and self.debug_mode):
            if hasattr(self, 'input_token_counts') and self.input_token_counts:
                self.logger.info("Offline scenario: Generating token statistics and histograms before cleanup...")
                self._print_token_histograms()
        
        # Clear stored response arrays after LoadGen is done
        if hasattr(self, 'response_arrays'):
            self.logger.debug(f"Clearing {len(self.response_arrays)} stored response arrays")
            self.response_arrays.clear()
        self.logger.info("Cleaning up LoadGen client")
        self.is_running = False
    
    def _get_sampling_params(self):
        """Get sampling parameters based on test_mode."""
        if self.test_mode == "accuracy":
            temperature = self.accuracy_temperature if self.accuracy_temperature is not None else self.temperature
            top_k = self.accuracy_top_k if self.accuracy_top_k is not None else self.top_k
            top_p = self.accuracy_top_p if self.accuracy_top_p is not None else self.top_p
            self.logger.debug(f"_get_sampling_params() - Accuracy mode: temperature={temperature}, top_k={top_k}, top_p={top_p}")
        else:
            temperature = self.temperature
            top_k = self.top_k
            top_p = self.top_p
            self.logger.debug(f"_get_sampling_params() - Performance mode: temperature={temperature}, top_k={top_k}, top_p={top_p}")
        return temperature, top_k, top_p
    
    def _print_sampling_summary(self):
        """Print final summary of sampling parameters and max_tokens."""
        temperature, top_k, top_p = self._get_sampling_params()
        self.logger.info("=" * 60)
        self.logger.info("Sampling Parameters Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Test Mode: {self.test_mode}")
        self.logger.info(f"Max Tokens: {self.max_tokens}")
        self.logger.info(f"Temperature: {temperature}")
        self.logger.info(f"Top-K: {top_k}")
        self.logger.info(f"Top-P: {top_p}")
        self.logger.info("=" * 60)
    
    def _track_token_stats(self, input_token_count: int, output_token_count: int):
        """Track token statistics for histograms."""
        if input_token_count > 0:
            self.input_token_counts.append(input_token_count)
            self.output_token_counts.append(output_token_count)
            ratio = input_token_count / output_token_count if output_token_count > 0 else 0.0
            self.token_ratios.append(ratio)
    
    def _print_token_histograms(self):
        """Print histograms of input tokens, output tokens, and token ratios."""
        # Enable if either debug_mode or print_token_stats is True
        if not self.input_token_counts:
            self.logger.warning(f"No token statistics collected. input_token_counts is empty. "
                              f"debug_mode={self.debug_mode}, print_token_stats={self.print_token_stats}")
            return
        
        if not self.debug_mode and not self.print_token_stats:
            return
        
        self.logger.info(f"Printing token histograms: collected {len(self.input_token_counts)} samples, "
                        f"debug_mode={self.debug_mode}, print_token_stats={self.print_token_stats}")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            
            # Use visualizations_output_dir if available, otherwise construct from output_dir
            if self.config and 'visualizations_output_dir' in self.config:
                output_dir = Path(self.config['visualizations_output_dir'])
            else:
                # Fallback: construct visualizations directory from output_dir
                base_output_dir = Path(self.config.get('output_dir', './harness_output')) if self.config else Path('./harness_output')
                output_dir = base_output_dir / 'visualizations'
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Input token histogram
            axes[0].hist(self.input_token_counts, bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Input Tokens')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Input Token Distribution\n(Mean: {np.mean(self.input_token_counts):.1f}, Median: {np.median(self.input_token_counts):.1f})')
            axes[0].grid(True, alpha=0.3)
            
            # Output token histogram
            axes[1].hist(self.output_token_counts, bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[1].set_xlabel('Output Tokens')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Output Token Distribution\n(Mean: {np.mean(self.output_token_counts):.1f}, Median: {np.median(self.output_token_counts):.1f})')
            axes[1].grid(True, alpha=0.3)
            
            # Token ratio histogram
            axes[2].hist(self.token_ratios, bins=50, edgecolor='black', alpha=0.7, color='orange')
            axes[2].set_xlabel('Input/Output Token Ratio')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'Token Ratio Distribution\n(Mean: {np.mean(self.token_ratios):.4f}, Median: {np.median(self.token_ratios):.4f})')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            histogram_path = output_dir / 'token_statistics_histograms.png'
            plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info("=" * 60)
            self.logger.info("Token Statistics Summary")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Queries: {len(self.input_token_counts)}")
            self.logger.info(f"Input Tokens - Mean: {np.mean(self.input_token_counts):.1f}, Median: {np.median(self.input_token_counts):.1f}, "
                           f"Min: {np.min(self.input_token_counts)}, Max: {np.max(self.input_token_counts)}")
            self.logger.info(f"Output Tokens - Mean: {np.mean(self.output_token_counts):.1f}, Median: {np.median(self.output_token_counts):.1f}, "
                           f"Min: {np.min(self.output_token_counts)}, Max: {np.max(self.output_token_counts)}")
            self.logger.info(f"Token Ratio (input/output) - Mean: {np.mean(self.token_ratios):.4f}, Median: {np.median(self.token_ratios):.4f}, "
                           f"Min: {np.min(self.token_ratios):.4f}, Max: {np.max(self.token_ratios):.4f}")
            self.logger.info(f"Histograms saved to: {histogram_path}")
            self.logger.info("=" * 60)
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping histogram generation")
        except Exception as e:
            self.logger.warning(f"Error generating histograms: {e}")
