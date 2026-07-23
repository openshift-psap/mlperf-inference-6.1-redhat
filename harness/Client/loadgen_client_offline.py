# ============================================================================
# loadgen_offline_client.py
# -------------------------
# MLPerf LoadGen Offline scenario client implementation
# ============================================================================

import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

# Import MLPerf Loadgen
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    import sys
    sys.exit(1)

# Import base LoadGenClient
from .loadgen_client import LoadGenClient


class LoadGenOfflineClient(LoadGenClient):
    """LoadGen client for Offline scenario."""
    
    def __init__(self, *args, **kwargs):
        # Force scenario to Offline
        kwargs['scenario'] = 'Offline'
        super().__init__(*args, **kwargs)
        self.batch_counter = 0
    
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """Process queries in batches for offline scenario."""
        total_samples = len(query_samples)
        
        # Determine sampling parameters based on test_mode
        temperature, top_k, top_p = self._get_sampling_params()
        
        if self.offline_back_to_back:
            # Send requests asynchronously (fire-and-forget, process responses as they arrive)
            self.logger.info("=" * 80)
            self.logger.info(f"OFFLINE SCENARIO: Processing {total_samples} queries asynchronously")
            self.logger.info(f"Mode: Individual API requests (one prompt per request, no batching)")
            self.logger.info(f"Total samples: {total_samples}")
            self.logger.info(f"Max concurrent requests: {self.offline_async_concurrency}")
            self.logger.info(f"offline_back_to_back flag: {self.offline_back_to_back}")
            self.logger.info("=" * 80)
            
            if not (self.api_server_url or self.api_server_urls):
                self.logger.warning("Local model processing not yet implemented")
                self._send_error_responses(query_samples)
                return
            
            # Use ThreadPoolExecutor for async request processing
            # This allows us to send multiple requests concurrently and process responses as they arrive
            processed_samples = 0
            completed_samples = 0
            failed_samples = 0
            
            # Thread-safe counter for progress tracking
            progress_lock = threading.Lock()
            
            def process_single_async(q_sample: 'lg.QuerySample') -> tuple:
                """Process a single query asynchronously. Returns (success, q_sample, response_data)."""
                nonlocal completed_samples, failed_samples
                try:
                    # This will block until the request completes, but we run multiple in parallel
                    # Get response data from worker thread without calling LoadGen callback
                    response_data = self._process_api_single(q_sample, temperature, top_k, top_p)
                    with progress_lock:
                        completed_samples += 1
                        if completed_samples % 100 == 0:
                            self.logger.info(f"Completed {completed_samples}/{total_samples} responses (async)")
                    return (True, q_sample, response_data)
                except Exception as e:
                    self.logger.error(f"Error processing query {q_sample.id}: {e}", exc_info=True)
                    with progress_lock:
                        failed_samples += 1
                    return (False, q_sample, None)
            
            # Submit all requests to thread pool
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.offline_async_concurrency) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(process_single_async, q_sample): q_sample 
                    for q_sample in query_samples
                }
                
                # Process futures as they complete (responses arrive)
                # This allows us to handle responses as soon as they're ready
                # IMPORTANT: Call lg.QuerySamplesComplete from main thread only
                for future in as_completed(future_to_sample):
                    q_sample = future_to_sample[future]
                    try:
                        success, q_sample_result, response_data = future.result()

                        if success and response_data is not None:
                            # Call LoadGen callback from main thread (thread-safe)
                            query_id = response_data['query_id']
                            output_data_ptr = response_data['output_data_ptr']
                            output_data_size = response_data['output_data_size']
                            n_tokens = response_data['n_tokens']

                            # Create response for LoadGen
                            response_array = [
                                lg.QuerySampleResponse(
                                    query_id,
                                    output_data_ptr,
                                    output_data_size,
                                    n_tokens
                                )
                            ]

                            # Report completion to LoadGen from main thread
                            lg.QuerySamplesComplete(response_array)
                            self.logger.debug(f"Query {query_id}: {n_tokens} tokens")
                        else:
                            # Send error response from main thread
                            self._send_error_responses([q_sample_result])

                        processed_samples += 1
                        if processed_samples % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = processed_samples / elapsed if elapsed > 0 else 0
                            self.logger.info(f"Processed {processed_samples}/{total_samples} requests "
                                           f"({rate:.1f} req/s)")
                    except Exception as e:
                        self.logger.error(f"Future exception for query {q_sample.id}: {e}")
                        self._send_error_responses([q_sample])
                        failed_samples += 1
            
            # All requests have been submitted and responses processed
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info(f"All queries completed: {completed_samples} succeeded, {failed_samples} failed")
            self.logger.info(f"Total time: {total_time:.2f}s, Avg rate: {total_samples/total_time:.2f} req/s")
            self.logger.info("=" * 80)
            
            # Print token histograms if enabled
            if self.print_token_stats or self.debug_mode:
                self._print_token_histograms()
        else:
            # Batch processing (original behavior)
            num_batches = (total_samples + self.batch_size - 1) // self.batch_size
            
            self.logger.info("=" * 80)
            self.logger.info(f"OFFLINE SCENARIO: Processing {total_samples} queries in {num_batches} batches")
            self.logger.info(f"Batch size: {self.batch_size}, Total samples: {total_samples}")
            self.logger.info(f"offline_back_to_back flag: {self.offline_back_to_back} (using batch mode)")
            self.logger.info("=" * 80)
            
            processed_samples = 0
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, total_samples)
                batch = query_samples[start:end]
                batch_size = len(batch)
                
                self.logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({batch_size} samples)")
                
                try:
                    if self.api_server_url:
                        self._process_api_batch(batch, temperature, top_k, top_p)
                        processed_samples += batch_size
                        self.logger.info(f"✓ Batch {batch_idx + 1}/{num_batches} completed ({processed_samples}/{total_samples} samples processed)")
                    else:
                        self.logger.warning("Local model processing not yet implemented")
                        # TODO: Implement local model processing
                        self._send_error_responses(batch)
                        processed_samples += batch_size
                    
                    self.batch_counter += 1
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx + 1}/{num_batches}: {e}", exc_info=True)
                    self._send_error_responses(batch)
                    processed_samples += batch_size
            
            self.logger.info("=" * 80)
            self.logger.info(f"All batches completed: {processed_samples}/{total_samples} samples processed")
            self.logger.info("=" * 80)
            
            # Print token histograms if enabled
            if self.print_token_stats or self.debug_mode:
                self._print_token_histograms()
    
    def _process_api_single(self, q_sample: 'lg.QuerySample', temperature: float, top_k: int, top_p: float) -> Dict[str, Any]:
        """
        Process a single query via API (for back-to-back mode).

        This method sends ONE prompt in ONE API request - no batching.
        Each call to this method results in a separate HTTP request to the API server.

        Returns:
            Dictionary with response data for LoadGen (query_id, output_data_ptr, output_data_size, n_tokens)
        """
        # Log that we're processing a single request (not a batch)
        self.logger.debug(f"[BACK-TO-BACK MODE] Processing single query {q_sample.id} (index {q_sample.index}) - individual API request")
        
        # Get input IDs from dataset
        input_ids = self.dataset.input_ids[q_sample.index]
        
        # Check if using SGLang with input_ids
        if self.use_input_ids:
            # SGLang format: send input_ids directly
            # Get server URL (with load balancing if enabled)
            server_url = self._get_next_server_url()
            endpoint = f"{server_url}{self.sglang_endpoint}"
            api_payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                }
            }
            
            self.logger.debug(f"Sending SGLang request to {endpoint} for query {q_sample.id}")
            response = self._send_request_with_retry(endpoint, api_payload, server_url)

            api_result = response.json()
            output_ids = api_result.get("output_ids", [])
            output_text = api_result.get("text", "")

            # Process response and return data (don't call LoadGen callback)
            return self._process_sglang_response(q_sample.id, q_sample.index, output_ids, output_text)
        else:
            # Standard format: use text_input directly if available (e.g., for gpt-oss-120b with vLLM)
            # Otherwise decode from input_ids
            if (hasattr(self.dataset, 'input') and 
                len(self.dataset.input) > q_sample.index and 
                self.dataset.input[q_sample.index]):
                # Use text_input directly (no detokenization needed)
                text_prompt = self.dataset.input[q_sample.index]
                self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id} (index {q_sample.index}): Using text_input directly from dataset")
            elif self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id} (index {q_sample.index}): Decoded from input_ids using tokenizer (length: {len(text_prompt)} chars)")
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens for query {q_sample.id}: {e}")
                    text_prompt = " ".join([str(t) for t in input_ids])
                    self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id}: Fallback to string representation of input_ids")
            else:
                text_prompt = " ".join([str(t) for t in input_ids])
                self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id}: No tokenizer available, using string representation of input_ids")
            
            # Log the text prompt (first 200 chars) for debugging
            text_preview = text_prompt[:200] + "..." if len(text_prompt) > 200 else text_prompt
            self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id}: Text prompt preview: {text_preview}")
            self.logger.debug(f"LoadGenOfflineClient._process_api_single() - Query {q_sample.id}: Full text prompt length: {len(text_prompt)} chars, input_ids length: {len(input_ids)} tokens")
            
            # Get server URL (with load balancing if enabled)
            server_url = self._get_next_server_url()
            endpoints = self._get_endpoints_for_url(server_url)
            
            # Determine endpoint based on endpoint_type
            if self.endpoint_type == 'chat_completions':
                endpoint = endpoints['chat_completions']
                api_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": text_prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    "return_token_ids": True  # Request token IDs directly from API
                }
            else:
                endpoint = endpoints['completions']
                api_payload = {
                    "model": self.model_name,
                    "prompt": text_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stream": False,
                    "return_token_ids": True  # Request token IDs directly from API
                }

            self.logger.debug(f"Sending API request to {endpoint} for query {q_sample.id} and temperture {temperature}, top_k {top_k}, top_p {top_p}")
            response = self._send_request_with_retry(endpoint, api_payload, server_url)

            api_result = response.json()

            # Extract response
            choice = api_result.get("choices", [{}])[0]

            # Try to get token_ids directly from API response first
            token_ids = choice.get("token_ids", None)

            if token_ids is None:
                # Fallback: extract text and re-encode
                self.logger.debug(f"token_ids not found in API response for query {q_sample.id}, falling back to text encoding")
                if self.endpoint_type == 'chat_completions':
                    text_response = choice.get("message", {}).get("content", "")
                else:
                    text_response = choice.get("text", "")

                # Convert to token IDs as fallback
                if self.tokenizer:
                    try:
                        token_ids = self.tokenizer.encode(text_response, add_special_tokens=False)
                    except Exception as e:
                        self.logger.warning(f"Error encoding response: {e}")
                        token_ids = []
                else:
                    token_ids = []
            else:
                # Got token_ids directly from API - also extract text for logging
                self.logger.debug(f"Using token_ids from API response for query {q_sample.id}: {len(token_ids)} tokens")
                if self.endpoint_type == 'chat_completions':
                    text_response = choice.get("message", {}).get("content", "")
                else:
                    text_response = choice.get("text", "")

            # Process response and return data (don't call LoadGen callback)
            return self._process_single_response(q_sample.id, q_sample.index, token_ids, text_response, text_prompt)
    
    def _process_api_batch(self, batch: List['lg.QuerySample'], temperature: float, top_k: int, top_p: float) -> None:
        """Process a batch via API."""
        batch_size = len(batch)
        self.logger.debug(f"Processing API batch with {batch_size} samples")
        
        # Check if using SGLang with input_ids
        if self.use_input_ids:
            # SGLang format: send each request individually (SGLang handles batching internally)
            # In batch mode (non-async), we call LoadGen immediately from this thread (safe)
            for q_sample in batch:
                response_data = self._process_api_single(q_sample, temperature, top_k, top_p)
                if response_data:
                    # Create and send LoadGen response
                    response_array = [
                        lg.QuerySampleResponse(
                            response_data['query_id'],
                            response_data['output_data_ptr'],
                            response_data['output_data_size'],
                            response_data['n_tokens']
                        )
                    ]
                    lg.QuerySamplesComplete(response_array)
            return
        
        # Standard format: prepare text prompts
        text_prompts = []
        original_query_ids = []
        original_query_indexes = []
        
        for q_sample in batch:
            original_query_ids.append(q_sample.id)
            original_query_indexes.append(q_sample.index)
            
            # Get input IDs from dataset
            input_ids = self.dataset.input_ids[q_sample.index]
            
            # Use text_input directly if available (e.g., for gpt-oss-120b with vLLM)
            # Otherwise decode from input_ids
            if (hasattr(self.dataset, 'input') and 
                len(self.dataset.input) > q_sample.index and 
                self.dataset.input[q_sample.index]):
                # Use text_input directly (no detokenization needed)
                text_prompt = self.dataset.input[q_sample.index]
                text_prompts.append(text_prompt)
                self.logger.debug(f"LoadGenOfflineClient._process_api_batch() - Query {q_sample.id} (index {q_sample.index}): Using text_input directly from dataset (length: {len(text_prompt)} chars)")
            elif self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    text_prompts.append(text_prompt)
                    self.logger.debug(f"LoadGenOfflineClient._process_api_batch() - Query {q_sample.id} (index {q_sample.index}): Decoded from input_ids using tokenizer (length: {len(text_prompt)} chars)")
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens for query {q_sample.id}: {e}")
                    text_prompt = " ".join([str(t) for t in input_ids])
                    text_prompts.append(text_prompt)
                    self.logger.debug(f"LoadGenOfflineClient._process_api_batch() - Query {q_sample.id}: Fallback to string representation of input_ids")
            else:
                text_prompt = " ".join([str(t) for t in input_ids])
                text_prompts.append(text_prompt)
                self.logger.debug(f"LoadGenOfflineClient._process_api_batch() - Query {q_sample.id}: No tokenizer available, using string representation of input_ids")
            
            # Log first prompt in batch for debugging (to avoid too much output)
            if q_sample.index == batch[0].index:
                text_preview = text_prompt[:200] + "..." if len(text_prompt) > 200 else text_prompt
                self.logger.debug(f"LoadGenOfflineClient._process_api_batch() - First prompt in batch (query {q_sample.id}): {text_preview}")
        
        self.logger.debug(f"Prepared {len(text_prompts)} prompts for API batch")
        
        # Get server URL (with load balancing if enabled)
        server_url = self._get_next_server_url()
        endpoints = self._get_endpoints_for_url(server_url)
        
        # Determine endpoint based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint = endpoints['chat_completions']
            # Format for chat completions API - handle batch requests
            # For chat completions, we need to send each prompt separately or use array format
            # Most APIs support array format for batch processing
            if len(text_prompts) == 1:
                # Single prompt
                api_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": text_prompts[0]}],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    "return_token_ids": True  # Request token IDs directly from API
                }
            else:
                # Batch: send as array of messages arrays
                # Note: Some APIs may require separate requests for batch
                api_payload = {
                    "model": self.model_name,
                    "messages": [[{"role": "user", "content": prompt}] for prompt in text_prompts],
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    "return_token_ids": True  # Request token IDs directly from API
                }
        else:
            endpoint = endpoints['completions']
            # Format for completions API
            api_payload = {
                "model": self.model_name,
                "prompt": text_prompts,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stream": False,
                "return_token_ids": True  # Request token IDs directly from API
            }
        
        self.logger.debug(f"Sending API batch request to {endpoint} with {len(text_prompts)} prompts, max tokens {self.max_tokens}, temperature {temperature}, top_k {top_k}, top_p {top_p}")
        response = self._send_request_with_retry(endpoint, api_payload, server_url)
        
        api_result = response.json()
        self.logger.debug(f"Received API response with {len(api_result.get('choices', []))} choices")
        
        # Extract choices based on endpoint type
        if self.endpoint_type == 'chat_completions':
            # Chat completions: handle batch vs single response
            all_choices = api_result.get("choices", [])
            if isinstance(all_choices, list) and len(all_choices) > 0:
                # Check if first element is a list (batch) or dict (single)
                if isinstance(all_choices[0], list):
                    # Batch response: flatten list of lists
                    choices = []
                    for choice_group in all_choices:
                        for choice in choice_group:
                            if isinstance(choice, dict) and "message" in choice:
                                choices.append({"text": choice["message"].get("content", "")})
                            else:
                                choices.append(choice)
                else:
                    # Single response or array of choices
                    choices = []
                    for choice in all_choices:
                        if isinstance(choice, dict) and "message" in choice:
                            choices.append({"text": choice["message"].get("content", "")})
                        else:
                            choices.append(choice)
            else:
                choices = []
        else:
            # Completions endpoint returns choices with text
            choices = api_result.get("choices", [])
        
        # Process responses
        self._process_api_responses(choices, original_query_ids, original_query_indexes, text_prompts)
    
    def _process_sglang_response(self, query_id: int, query_index: int, output_ids: List[int], output_text: str) -> Dict[str, Any]:
        """Process SGLang response (already has token IDs).

        Returns:
            Dictionary with response data for LoadGen (query_id, output_data_ptr, output_data_size, n_tokens)
        """
        token_count = len(output_ids)
        
        # Get input token count for logging
        input_token_count = 0
        if hasattr(self, 'dataset') and self.dataset and query_index < len(self.dataset.input_ids):
            input_token_count = len(self.dataset.input_ids[query_index])
        
        # Store results in accuracy mode (matching offline_sut.py format)
        if self.test_mode == "accuracy" and self.results is not None:
            # Get metadata from response if available
            metadata = {}
            # Store in same format as offline_sut.py: process_single_query
            self.results[query_id] = {
                "output_ids": output_ids,
                "output_text": output_text,
                "metadata": metadata
            }
        
        # Debug mode: print query, text response, token counts and ratio
        if self.debug_mode and self.test_mode == "accuracy":
            # Get prompt text if available
            prompt_text = None
            if hasattr(self, 'dataset') and self.dataset:
                if query_index < len(self.dataset.input) and self.dataset.input[query_index]:
                    prompt_text = self.dataset.input[query_index]
                elif self.tokenizer and query_index < len(self.dataset.input_ids):
                    try:
                        prompt_text = self.tokenizer.decode(self.dataset.input_ids[query_index], skip_special_tokens=True)
                    except:
                        prompt_text = f"[Token IDs: {self.dataset.input_ids[query_index][:50]}...]"
            
            text_preview = output_text[:200] + "..." if len(output_text) > 200 else output_text
            prompt_preview = prompt_text[:200] + "..." if prompt_text and len(prompt_text) > 200 else (prompt_text or "N/A")
            
            # Calculate token ratio (input/output)
            token_ratio = (input_token_count / token_count) if token_count > 0 else 0.0
            
            # Track token statistics (always track if print_token_stats is enabled, or in debug mode)
            if self.print_token_stats or self.debug_mode:
                self._track_token_stats(input_token_count, token_count)
            
            # Debug mode: print query, text response, token counts and ratio
            if self.debug_mode and self.test_mode == "accuracy":
                self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
                if prompt_text:
                    self.logger.info(f"  Prompt: {prompt_preview}")
                self.logger.debug(f"  Text : {prompt_text}")
                self.logger.info(f"  Text Response: {text_preview}")
                self.logger.info(f"  Input Tokens: {input_token_count}")
                self.logger.info(f"  Output Tokens: {token_count}")
                self.logger.info(f"  Token Ratio (input/output): {token_ratio:.4f}")
        
        # Convert output_ids to numpy array for LoadGen
        # LoadGen expects int32 token IDs as a contiguous array
        if output_ids:
            token_array = np.ascontiguousarray(output_ids, dtype=np.int32)
            # CRITICAL: Keep array alive to prevent garbage collection before LoadGen reads it
            self.response_arrays[query_id] = token_array
            output_data_ptr = token_array.ctypes.data
            output_data_size = token_array.nbytes
            n_tokens = len(output_ids)
        else:
            # Empty response
            token_array = np.array([], dtype=np.int32)
            output_data_ptr = 0
            output_data_size = 0
            n_tokens = 0

        # Return response data for LoadGen (caller will invoke lg.QuerySamplesComplete from main thread)
        return {
            'query_id': query_id,
            'output_data_ptr': output_data_ptr,
            'output_data_size': output_data_size,
            'n_tokens': n_tokens
        }
    
    def _process_single_response(self, query_id: int, query_index: int, token_ids: List[int], text_response: str, text_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single response.

        Returns:
            Dictionary with response data for LoadGen (query_id, output_data_ptr, output_data_size, n_tokens)
        """
        token_count = len(token_ids)
        
        # Get input token count for logging
        input_token_count = 0
        if hasattr(self, 'dataset') and self.dataset and query_index < len(self.dataset.input_ids):
            input_token_count = len(self.dataset.input_ids[query_index])
        
        # Store results in accuracy mode (matching offline_sut.py format)
        if self.test_mode == "accuracy" and self.results is not None:
            # Get metadata from response if available
            metadata = {}
            # Store in same format as offline_sut.py: process_single_query
            self.results[query_id] = {
                "output_ids": token_ids,
                "output_text": text_response,
                "metadata": metadata
            }
        
        # Track token statistics (always track if print_token_stats is enabled, or in debug mode)
        if self.print_token_stats or self.debug_mode:
            self._track_token_stats(input_token_count, token_count)
        
        # Debug mode: print query, text response, token counts and ratio
        if self.debug_mode and self.test_mode == "accuracy":
            query_preview = text_prompt[:200] + "..." if text_prompt and len(text_prompt) > 200 else (text_prompt or "N/A")
            text_preview = text_response[:200] + "..." if len(text_response) > 200 else text_response
            
            # Calculate token ratio (input/output)
            token_ratio = (input_token_count / token_count) if token_count > 0 else 0.0
            
            self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
            self.logger.info(f"  Prompt: {query_preview}")
            self.logger.info(f"  Text Response: {text_preview}")
            self.logger.info(f"  Input Tokens: {input_token_count}")
            self.logger.info(f"  Output Tokens: {token_count}")
            self.logger.info(f"  Token Ratio (input/output): {token_ratio:.4f}")
        
        # Convert token_ids to numpy array for LoadGen
        # LoadGen expects int32 token IDs as a contiguous array
        if token_ids:
            token_array = np.ascontiguousarray(token_ids, dtype=np.int32)
            # CRITICAL: Keep array alive to prevent garbage collection before LoadGen reads it
            self.response_arrays[query_id] = token_array
            output_data_ptr = token_array.ctypes.data
            output_data_size = token_array.nbytes
            n_tokens = len(token_ids)
        else:
            # Empty response
            token_array = np.array([], dtype=np.int32)
            output_data_ptr = 0
            output_data_size = 0
            n_tokens = 0

        # Return response data for LoadGen (caller will invoke lg.QuerySamplesComplete from main thread)
        return {
            'query_id': query_id,
            'output_data_ptr': output_data_ptr,
            'output_data_size': output_data_size,
            'n_tokens': n_tokens
        }
    
    def _process_api_responses(self, choices: List[Dict], query_ids: List[int], query_indexes: List[int], text_prompts: Optional[List[str]] = None) -> None:
        """Process API responses and send to Loadgen."""
        self.logger.debug(f"Processing {len(choices)} API responses for {len(query_ids)} queries")

        responses = []
        for i, choice in enumerate(choices):
            if i >= len(query_ids):
                self.logger.warning(f"More choices than query IDs: {len(choices)} choices, {len(query_ids)} query IDs")
                break

            query_id = query_ids[i]
            query_index = query_indexes[i]

            # Try to get token_ids directly from API response first
            token_ids = choice.get("token_ids", None)

            if token_ids is None:
                # Fallback: extract text and re-encode
                self.logger.debug(f"token_ids not found in API response for query {query_id}, falling back to text encoding")
                text_response = choice.get("text", "")

                # Convert back to token IDs
                if self.tokenizer:
                    try:
                        token_ids = self.tokenizer.encode(text_response, add_special_tokens=False)
                    except Exception as e:
                        self.logger.warning(f"Error encoding response for query {query_id}: {e}")
                        token_ids = [1, 2, 3]  # Fallback
                else:
                    token_ids = [1, 2, 3]  # Fallback
            else:
                # Got token_ids directly from API
                self.logger.debug(f"Using token_ids from API response for query {query_id}: {len(token_ids)} tokens")
                text_response = choice.get("text", "")

            # Get the original query/prompt if available
            query_prompt = None
            if text_prompts and i < len(text_prompts):
                query_prompt = text_prompts[i]
            
            token_count = len(token_ids)
            
            # Get input token count for logging
            input_token_count = 0
            if hasattr(self, 'dataset') and self.dataset and query_index < len(self.dataset.input_ids):
                input_token_count = len(self.dataset.input_ids[query_index])
            
            # Store results in accuracy mode (matching offline_sut.py format)
            if self.test_mode == "accuracy" and self.results is not None:
                # Get metadata from response if available
                metadata = {}
                # Store in same format as offline_sut.py: process_single_query
                self.results[query_id] = {
                    "output_ids": token_ids,
                    "output_text": text_response,
                    "metadata": metadata
                }
            
            # Track token statistics (always track if print_token_stats is enabled, or in debug mode)
            if self.print_token_stats or self.debug_mode:
                self._track_token_stats(input_token_count, token_count)
            
            # Debug mode: print query, text response, token counts and ratio
            if self.debug_mode and self.test_mode == "accuracy":
                # Truncate text for display (first 200 chars)
                query_preview = query_prompt[:200] + "..." if query_prompt and len(query_prompt) > 200 else (query_prompt or "N/A")
                text_preview = text_response[:200] + "..." if len(text_response) > 200 else text_response
                
                # Calculate token ratio (input/output)
                token_ratio = (input_token_count / token_count) if token_count > 0 else 0.0
                
                self.logger.info(f"[DEBUG] Query {query_id} (index {query_index}):")
                self.logger.info(f"  Query: {query_preview}")
                self.logger.info(f"  Text Response: {text_preview}")
                self.logger.info(f"  Input Tokens: {input_token_count}")
                self.logger.info(f"  Output Tokens: {token_count}")
                self.logger.info(f"  Token Ratio (input/output): {token_ratio:.4f}")
            
            # Convert token_ids to numpy array for LoadGen
            # LoadGen expects int32 token IDs as a contiguous array
            if token_ids:
                token_array = np.ascontiguousarray(token_ids, dtype=np.int32)
                # CRITICAL: Keep array alive to prevent garbage collection before LoadGen reads it
                self.response_arrays[query_id] = token_array
                output_data_ptr = token_array.ctypes.data
                output_data_size = token_array.nbytes
                n_tokens = len(token_ids)
            else:
                # Empty response
                token_array = np.array([], dtype=np.int32)
                output_data_ptr = 0
                output_data_size = 0
                n_tokens = 0
            
            # Create response for LoadGen with token count
            response = lg.QuerySampleResponse(
                query_id,
                output_data_ptr,
                output_data_size,
                n_tokens  # Number of output tokens for tokens/sec metric
            )
            responses.append(response)
            self.logger.debug(f"Query {query_id} (index {query_index}): {n_tokens} tokens")
        
        # Send all responses to LoadGen
        if responses:
            lg.QuerySamplesComplete(responses)
            self.logger.debug(f"Sent {len(responses)} responses to LoadGen")
        else:
            self.logger.warning(f"No responses to send for batch (expected {len(query_ids)} responses)")
    
    def _send_error_responses(self, batch: List['lg.QuerySample']) -> None:
        """Send error responses for a batch."""
        for q_sample in batch:
            response = lg.QuerySampleResponse(q_sample.id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
    
    def flush_queries(self) -> None:
        """Flush queries (no-op for offline scenario)."""
        self.logger.debug("Flush queries called (no-op for offline)")
