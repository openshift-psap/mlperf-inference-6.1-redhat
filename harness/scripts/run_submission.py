#!/usr/bin/env python3
"""
MLPerf Inference Harness Test Runner
=====================================
Python script to generate and execute MLPerf harness test commands.

This script replaces the bash script with a cleaner Python implementation
that generates bash commands for execution.

Sample Usage:
-------------
# Run all Server tests
python3 run_submission.py --scenario Server --server-target-qps 3 run-server

# Run all Offline tests
python3 run_submission.py --scenario Offline run-offline

# Run all tests for both scenarios
python3 run_submission.py --server-target-qps 3 run-all

# Run specific test type
python3 run_submission.py --scenario Server --server-target-qps 3 run-performance
python3 run_submission.py --scenario Server --server-target-qps 3 run-accuracy

# Run compliance tests (runs both TEST07 and TEST09 by default)
python3 run_submission.py --scenario Offline run-compliance
python3 run_submission.py run-compliance TEST07  # Run only TEST07
python3 run_submission.py --scenario Server --server-target-qps 3 run-compliance TEST09

# Dry run to see commands without executing
python3 run_submission.py --dry-run run-server

# Generate bash script
python3 run_submission.py --print-bash run-server > run_tests.sh
python3 run_submission.py --print-bash --scenario Offline run-compliance > compliance_tests.sh

# Run without MLflow integration
python3 run_submission.py --no-mlflow --scenario Offline run-offline
python3 run_submission.py --no-mlflow --scenario Server --server-target-qps 3 run-server

"""

import argparse
import os
import sys
import subprocess
import shutil
import resource
from pathlib import Path
from typing import List, Optional, Dict, Tuple


class HarnessRunner:
    """Main class for running MLPerf harness tests."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.harness_dir = self.script_dir.parent
        
        # Configuration with defaults
        self.config = {
            'dataset_dir': os.environ.get('DATASET_DIR', ''),
            'perf_dataset': os.environ.get('PERF_DATASET', ''),
            'acc_dataset': os.environ.get('ACC_DATASET', ''),
            'compliance_dataset': os.environ.get('COMPLIANCE_DATASET', ''),
            'output_dir': os.environ.get('OUTPUT_DIR', './harness_output'),
            'api_server_url': os.environ.get('API_SERVER_URL', ''),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID', ''),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY', ''),
            'mlflow_tracking_uri': os.environ.get('MLFLOW_TRACKING_URI', ''),
            'mlflow_experiment_name': os.environ.get('MLFLOW_EXPERIMENT_NAME', ''),
            'mlflow_user_tag': os.environ.get('MLFLOW_USER_TAG', ''),
            'hf_home': os.environ.get('HF_HOME', ''),
            'model_category': os.environ.get('MODEL_CATEGORY', 'gpt-oss-120b'),
            'model': os.environ.get('MODEL', 'openai/gpt-oss-120b'),
            'backend': os.environ.get('BACKEND', 'vllm'),
            'lg_model_name': os.environ.get('LG_MODEL_NAME', 'gpt-oss-120b'),
            'scenario': os.environ.get('SCENARIO', 'Server'),
            'server_target_qps': os.environ.get('SERVER_TARGET_QPS', '3'),
            'server_target_qps_set': False,
            'compliance_test': os.environ.get('COMPLIANCE_TEST', 'TEST07'),
            'audit_config_src': os.environ.get('AUDIT_CONFIG_SRC', ''),
            'audit_override_conf': os.environ.get('AUDIT_OVERRIDE_CONF', 'audit-override.cfg'),
            'user_conf': os.environ.get('USER_CONF', ''),
            'dry_run': False,
            'print_bash': False,
            'no_mlflow': False,
        }
        
        # Derive dataset paths if dataset_dir is set
        if self.config['dataset_dir']:
            if not self.config['perf_dataset']:
                self.config['perf_dataset'] = str(Path(self.config['dataset_dir']) / 'perf' / 'perf_eval_ref.parquet')
            if not self.config['acc_dataset']:
                self.config['acc_dataset'] = str(Path(self.config['dataset_dir']) / 'acc' / 'acc_eval_ref.parquet')
            if not self.config['compliance_dataset']:
                self.config['compliance_dataset'] = str(Path(self.config['dataset_dir']) / 'acc' / 'acc_eval_compliance_gpqa.parquet')
    
    def parse_args(self, args: List[str]) -> Tuple[str, List[str]]:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Run MLPerf inference harness tests',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all Server tests
  %(prog)s --scenario Server --server-target-qps 3 run-server

  # Run all Offline tests
  %(prog)s --scenario Offline run-offline

  # Run all tests for both scenarios
  %(prog)s --server-target-qps 3 run-all

  # Run compliance tests (runs both TEST07 and TEST09 by default)
  %(prog)s run-compliance
  %(prog)s --scenario Offline run-compliance
  
  # Run specific compliance test
  %(prog)s run-compliance TEST07
  %(prog)s --scenario Offline run-compliance TEST09

  # Dry run to see commands
  %(prog)s --dry-run run-server

  # Generate bash script
  %(prog)s --print-bash run-server > run_tests.sh
  %(prog)s --print-bash --scenario Offline run-compliance > compliance_tests.sh

            """
        )
        
        # Configuration options
        parser.add_argument('--dataset-dir', help='Dataset directory')
        parser.add_argument('--perf-dataset', help='Performance dataset path')
        parser.add_argument('--acc-dataset', help='Accuracy dataset path')
        parser.add_argument('--compliance-dataset', help='Compliance dataset path')
        parser.add_argument('--output-dir', help='Output directory (default: ./harness_output)')
        parser.add_argument('--api-server-url', help='API server URL')
        parser.add_argument('--aws-access-key-id', help='AWS access key ID')
        parser.add_argument('--aws-secret-access-key', help='AWS secret access key')
        parser.add_argument('--mlflow-tracking-uri', help='MLflow tracking URI')
        parser.add_argument('--mlflow-experiment-name', help='MLflow experiment name')
        parser.add_argument('--tag', '--mlflow-tag', dest='mlflow_user_tag', help='Additional MLflow tag')
        parser.add_argument('--hf-home', help='HuggingFace home directory')
        parser.add_argument('--scenario', choices=['Server', 'Offline'], help='Scenario (default: Server)')
        parser.add_argument('--server-target-qps', help='Target QPS for Server scenario')
        parser.add_argument('--compliance-test', help='Compliance test name (default: TEST07)')
        parser.add_argument('--audit-config', dest='audit_config_src', help='Path to audit.config file')
        parser.add_argument('--audit-override-conf', help='Path to audit-override.cfg (default: audit-override.cfg)')
        parser.add_argument('--user-conf', help='User config file for performance/accuracy tests')
        parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
        parser.add_argument('--print-bash', action='store_true', help='Print bash script with environment variables and commands')
        parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow integration (skip all MLflow arguments)')
        
        # Commands
        parser.add_argument('command', nargs='?', choices=['run-server', 'run-offline', 'run-all', 
                          'run-performance', 'run-accuracy', 'run-compliance'], 
                          default='run-server', help='Command to execute')
        parser.add_argument('command_args', nargs='*', help='Additional arguments for command')
        
        parsed = parser.parse_args(args)
        
        # Update config from parsed arguments
        for key, value in vars(parsed).items():
            if value is not None and key not in ['command', 'command_args', 'dry_run', 'print_bash', 'no_mlflow']:
                config_key = key.replace('-', '_')
                if config_key in self.config:
                    self.config[config_key] = value
                    if key == 'server_target_qps':
                        self.config['server_target_qps_set'] = True
        
        self.config['dry_run'] = parsed.dry_run
        self.config['print_bash'] = parsed.print_bash
        self.config['no_mlflow'] = parsed.no_mlflow
        
        # Handle run-compliance command args (scenario and/or test)
        command = parsed.command
        if command == 'run-compliance' and parsed.command_args:
            for arg in parsed.command_args:
                if arg in ['Server', 'Offline']:
                    self.config['scenario'] = arg
                elif arg.startswith('TEST'):
                    self.config['compliance_test'] = arg
        
        return command, parsed.command_args
    
    def validate_config(self) -> bool:
        """Validate required configuration."""
        errors = []
        warnings = []
        
        if self.config['dry_run']:
            print("[DRY RUN] Validating configuration (lenient mode)...")
        else:
            print("Validating configuration...")
        
        # Check required environment variables
        required_vars = {
            'dataset_dir': 'DATASET_DIR',
            'api_server_url': 'API_SERVER_URL',
            'aws_access_key_id': 'AWS_ACCESS_KEY_ID',
            'aws_secret_access_key': 'AWS_SECRET_ACCESS_KEY',
        }
        
        # Add MLflow requirements only if not disabled
        if not self.config['no_mlflow']:
            required_vars['mlflow_tracking_uri'] = 'MLFLOW_TRACKING_URI'
            required_vars['mlflow_experiment_name'] = 'MLFLOW_EXPERIMENT_NAME'
        
        for config_key, env_var in required_vars.items():
            if not self.config[config_key]:
                if self.config['dry_run']:
                    print(f"[DRY RUN] NOTE: {env_var} would be required: <not set>")
                else:
                    errors.append(f"{env_var} environment variable is not set\n"
                                f"       Please set it via: export {env_var}=<value>\n"
                                f"       Or use: --{config_key.replace('_', '-')} <value>")
        
        # Check datasets
        for dataset_type, dataset_path in [('Performance', self.config['perf_dataset']),
                                           ('Accuracy', self.config['acc_dataset']),
                                           ('Compliance', self.config['compliance_dataset'])]:
            if dataset_path and not Path(dataset_path).exists():
                if self.config['dry_run']:
                    print(f"[DRY RUN] NOTE: {dataset_type} dataset would be checked: {dataset_path}")
                else:
                    warnings.append(f"{dataset_type} dataset not found: {dataset_path}")
        
        # Check harness_main.py
        harness_main = self.harness_dir / 'harness_main.py'
        if not harness_main.exists():
            if self.config['dry_run']:
                print(f"[DRY RUN] NOTE: harness_main.py would be checked: {harness_main}")
            else:
                errors.append(f"harness_main.py not found at {harness_main}")
        
        # Check server-target-qps for Server scenario
        if self.config['scenario'] == 'Server':
            if not self.config['server_target_qps_set'] or not self.config['server_target_qps']:
                if not self.config['dry_run']:
                    errors.append("--server-target-qps is required for Server scenario\n"
                                "       Please specify it via: --server-target-qps <value>\n"
                                "       Or set it via environment variable: export SERVER_TARGET_QPS=<value>")
        
        # Print errors and warnings
        if errors and not self.config['dry_run']:
            print("\nERROR: Found the following errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        if warnings and not self.config['dry_run']:
            print("\nWARNING: Found the following warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print("Continuing anyway...\n")
        elif not self.config['dry_run']:
            print("✓ Configuration validation passed\n")
        else:
            print("[DRY RUN] ✓ Configuration check completed\n")
        
        return True
    
    def find_audit_config(self, test_name: str) -> Optional[str]:
        """Find audit.config file for compliance test."""
        if self.config['audit_config_src']:
            return self.config['audit_config_src']
        
        # Try to find in compliance directory
        compliance_dir = self.harness_dir.parent / 'compliance'
        test_dir = compliance_dir / test_name / self.config['model_category']
        audit_config = test_dir / 'audit.config'
        if audit_config.exists():
            return str(audit_config)
        
        # Try generic path
        test_dir = compliance_dir / test_name
        audit_config = test_dir / 'audit.config'
        if audit_config.exists():
            return str(audit_config)
        
        return None
    
    def build_command(self, scenario: str, test_mode: str, dataset_path: str, 
                     output_subdir: str = '', description: str = '', tags: str = '',
                     audit_config_path: Optional[str] = None) -> List[str]:
        """Build the harness command."""
        # Build output directory
        # For compliance tests, use "compliance" directory instead of test_mode
        output_mode = 'compliance' if audit_config_path else test_mode
        output_dir = Path(self.config['output_dir']) / scenario.lower() / output_mode
        if output_subdir:
            output_dir = output_dir / output_subdir
        
        # Base command
        cmd = [
            'python3',
            str(self.harness_dir / 'harness_main.py'),
            '--model-category', self.config['model_category'],
            '--model', self.config['model'],
            '--dataset-path', dataset_path,
            '--backend', self.config['backend'],
            '--lg-model-name', self.config['lg_model_name'],
            '--test-mode', test_mode,
            '--api-server-url', self.config['api_server_url'],
            '--scenario', scenario,
            '--output-dir', str(output_dir),
        ]
        
        # Add MLflow arguments only if not disabled
        if not self.config['no_mlflow']:
            cmd.extend(['--mlflow-experiment-name', self.config['mlflow_experiment_name']])
            
            # Add MLflow tracking URI
            if self.config['mlflow_tracking_uri']:
                uri = self.config['mlflow_tracking_uri'].replace('http://', '').replace('https://', '')
                if ':' in uri:
                    host, port = uri.split(':', 1)
                    cmd.extend(['--mlflow-host', host])
                    cmd.extend(['--mlflow-port', port])
                else:
                    cmd.extend(['--mlflow-host', uri])
            
            # Add MLflow description
            if description:
                cmd.extend(['--mlflow-description', description])
            
            # Add MLflow tags (merge user tag with existing tags)
            final_tags = tags
            if self.config['mlflow_user_tag']:
                if final_tags:
                    final_tags = f"{final_tags},{self.config['mlflow_user_tag']}"
                else:
                    final_tags = self.config['mlflow_user_tag']
            
            if final_tags:
                cmd.extend(['--mlflow-tag', final_tags])
        
        # Add server-target-qps for Server scenario
        if scenario == 'Server':
            cmd.extend(['--server-target-qps', self.config['server_target_qps']])
        
        # Add user-conf for performance/accuracy tests (if not compliance)
        # Priority: explicit user_conf > scenario-specific defaults
        user_conf_to_use = None
        if audit_config_path:
            # For compliance tests, use audit-override.conf (handled below)
            pass
        elif self.config['user_conf']:
            # Use explicitly provided user_conf
            user_conf_to_use = self.config['user_conf']
        elif test_mode in ['performance', 'accuracy']:
            # Use scenario-specific defaults
            if scenario == 'Server':
                # Server performance and accuracy use default.conf
                default_conf = self.script_dir.parent / 'default.conf'
                if default_conf.exists():
                    user_conf_to_use = str(default_conf)
            elif scenario == 'Offline':
                if test_mode == 'performance':
                    # Offline performance uses offline.conf
                    offline_conf = self.script_dir.parent / 'offline.conf'
                    if offline_conf.exists():
                        user_conf_to_use = str(offline_conf)
                elif test_mode == 'accuracy':
                    # Offline accuracy uses default.conf
                    default_conf = self.script_dir.parent / 'default.conf'
                    if default_conf.exists():
                        user_conf_to_use = str(default_conf)
        
        if user_conf_to_use:
            if Path(user_conf_to_use).exists() or self.config['dry_run']:
                cmd.extend(['--user-conf', user_conf_to_use])
            else:
                print(f"WARNING: User config file not found: {user_conf_to_use}, skipping --user-conf")
        
        # Add audit-override.cfg for compliance tests (except TEST09 which uses default.conf)
        if audit_config_path:
            # For TEST09 (both Server and Offline), use default.conf instead of audit-override.conf
            if output_subdir.lower() == 'test09':
                default_conf = self.script_dir.parent / 'default.conf'
                if default_conf.exists() or self.config['dry_run']:
                    cmd.extend(['--user-conf', str(default_conf)])
            else:
                # For other compliance tests, use audit-override.conf
                if Path(self.config['audit_override_conf']).exists() or self.config['dry_run']:
                    cmd.extend(['--user-conf', self.config['audit_override_conf']])
                else:
                    print(f"WARNING: Audit override config file not found: {self.config['audit_override_conf']}, skipping --user-conf")
        
        # Add offline-specific flags for offline performance, accuracy, and compliance tests
        # Note: Compliance tests use test_mode='performance', so they will also get these flags
        if scenario == 'Offline' and test_mode in ['performance', 'accuracy']:
            cmd.extend(['--offline-back-to-back', '--offline-async-concurrency', '6396'])
        
        # Add audit config for compliance tests
        if audit_config_path:
            # Copy audit.config to harness directory
            audit_dest = self.harness_dir / 'audit.config'
            cmd.extend(['--audit-config', 'audit.config'])
            
            if not self.config['dry_run']:
                shutil.copy2(audit_config_path, audit_dest)
        
        return cmd
    
    def print_command(self, cmd: List[str], scenario: str, test_mode: str, 
                     dataset_path: str, output_dir: str):
        """Print command in a readable multi-line format."""
        print("==========================================")
        if self.config['dry_run']:
            print(f"[DRY RUN] Would run: {scenario} scenario, {test_mode} mode")
        else:
            print(f"Running: {scenario} scenario, {test_mode} mode")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
        
        if self.config['dry_run']:
            print("[DRY RUN] Command would be:")
        else:
            print("Command:")
        
        # Print command in multi-line format using the formatting method
        formatted = self._format_command_as_bash(cmd)
        print(f"  {formatted}")
        print("==========================================")
    
    def run_command(self, cmd: List[str]) -> bool:
        """Run the command."""
        if self.config['dry_run']:
            print("[DRY RUN] Command not executed")
            return True
        
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"✗ Error running command: {e}")
            return False
    
    def run_test(self, scenario: str, test_mode: str, dataset_path: str,
                output_subdir: str = '', description: str = '', tags: str = '',
                audit_config_path: Optional[str] = None) -> bool:
        """Run a single test."""
        cmd = self.build_command(scenario, test_mode, dataset_path, output_subdir,
                               description, tags, audit_config_path)
        
        # For compliance tests, use "compliance" directory instead of test_mode
        output_mode = 'compliance' if audit_config_path else test_mode
        output_dir = Path(self.config['output_dir']) / scenario.lower() / output_mode
        if output_subdir:
            output_dir = output_dir / output_subdir
        
        self.print_command(cmd, scenario, test_mode, dataset_path, str(output_dir))
        
        success = self.run_command(cmd)
        
        # Cleanup audit.config if it was copied
        if audit_config_path and not self.config['dry_run']:
            audit_dest = self.harness_dir / 'audit.config'
            if audit_dest.exists():
                audit_dest.unlink()
                print("Cleaned up audit.config from harness directory")
        
        if success:
            print(f"✓ {scenario} {test_mode} test completed successfully")
        else:
            print(f"✗ {scenario} {test_mode} test failed")
        
        return success
    
    def run_performance(self, scenario: str) -> bool:
        """Run performance test."""
        description = f"{scenario} Performance"
        tags = f"test_type:performance,scenario:{scenario}"
        
        if scenario == 'Server':
            description = f"{scenario} Performance QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        
        return self.run_test(scenario, 'performance', self.config['perf_dataset'],
                           description=description, tags=tags)
    
    def run_accuracy(self, scenario: str) -> bool:
        """Run accuracy test."""
        description = f"{scenario} Accuracy"
        tags = f"test_type:accuracy,scenario:{scenario}"
        
        if scenario == 'Server':
            description = f"{scenario} Accuracy QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        
        return self.run_test(scenario, 'accuracy', self.config['acc_dataset'],
                           description=description, tags=tags)
    
    def run_compliance(self, scenario: str, compliance_test: Optional[str] = None) -> bool:
        """Run compliance test(s)."""
        # If no specific test specified, run both TEST07 and TEST09
        if compliance_test is None:
            print(">>> Running compliance test TEST07...")
            if not self._run_single_compliance(scenario, 'TEST07'):
                return False
            print()
            
            print(">>> Running compliance test TEST09...")
            if not self._run_single_compliance(scenario, 'TEST09'):
                return False
            print()
            
            print(f"✓ All compliance tests for {scenario} scenario completed successfully")
            return True
        else:
            # Run single specified test
            return self._run_single_compliance(scenario, compliance_test)
    
    def _run_single_compliance(self, scenario: str, test_name: str) -> bool:
        """Run a single compliance test."""
        audit_config_path = self.find_audit_config(test_name)
        
        if not audit_config_path:
            print(f"Error: Could not find audit.config for {test_name}")
            return False
        
        # Select dataset based on test type
        if test_name == 'TEST07':
            dataset_path = self.config['compliance_dataset']
            if not dataset_path:
                print("Error: COMPLIANCE_DATASET not set (required for TEST07)")
                return False
        elif test_name == 'TEST09':
            dataset_path = self.config['perf_dataset']
            if not dataset_path:
                print("Error: PERF_DATASET not set (required for TEST09)")
                return False
        else:
            dataset_path = self.config['compliance_dataset']
            print(f"WARNING: Unknown compliance test {test_name}, using COMPLIANCE_DATASET")
        
        description = f"{scenario} Compliance {test_name}"
        tags = f"test_type:compliance,scenario:{scenario},compliance_test:{test_name}"
        
        if scenario == 'Server':
            description = f"{scenario} Compliance {test_name} QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        
        # Compliance tests use performance mode with audit.config, but output to "compliance" directory
        # Note: For offline TEST09, --user-conf default.conf is added in build_command
        return self.run_test(scenario, 'performance', dataset_path, test_name.lower(),
                           description, tags, audit_config_path)
    
    def generate_bash_script(self, command: str, command_args: List[str]) -> str:
        """Generate bash script with environment variables and commands."""
        lines = ['#!/bin/bash', '', '# Generated bash script for MLPerf harness tests', '']
        
        # Clean up audit.config at the beginning
        lines.append('# Cleanup audit.config if it exists')
        lines.append(f'rm -f "{self.harness_dir / "audit.config"}"')
        lines.append('')
        
        # Set ulimit for file descriptors
        lines.append('# Set ulimit for file descriptors')
        lines.append('ulimit -n 32768')
        lines.append('')
        
        # Export environment variables
        lines.append('# Environment Variables')
        env_vars = {
            'DATASET_DIR': self.config['dataset_dir'],
            'PERF_DATASET': self.config['perf_dataset'],
            'ACC_DATASET': self.config['acc_dataset'],
            'COMPLIANCE_DATASET': self.config['compliance_dataset'],
            'OUTPUT_DIR': self.config['output_dir'],
            'API_SERVER_URL': self.config['api_server_url'],
            'AWS_ACCESS_KEY_ID': self.config['aws_access_key_id'],
            'AWS_SECRET_ACCESS_KEY': self.config['aws_secret_access_key'],
            'HF_HOME': self.config['hf_home'],
            'MODEL_CATEGORY': self.config['model_category'],
            'MODEL': self.config['model'],
            'BACKEND': self.config['backend'],
            'LG_MODEL_NAME': self.config['lg_model_name'],
            'SCENARIO': self.config['scenario'],
            'SERVER_TARGET_QPS': self.config['server_target_qps'],
            'COMPLIANCE_TEST': self.config['compliance_test'],
            'AUDIT_CONFIG_SRC': self.config['audit_config_src'],
            'AUDIT_OVERRIDE_CONF': self.config['audit_override_conf'],
            'USER_CONF': self.config['user_conf'],
        }
        
        # Add MLflow environment variables only if not disabled
        if not self.config['no_mlflow']:
            env_vars['MLFLOW_TRACKING_URI'] = self.config['mlflow_tracking_uri']
            env_vars['MLFLOW_EXPERIMENT_NAME'] = self.config['mlflow_experiment_name']
            env_vars['MLFLOW_USER_TAG'] = self.config['mlflow_user_tag']
        
        for var_name, var_value in env_vars.items():
            if var_value:
                # Escape special characters in the value
                escaped_value = var_value.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
                lines.append(f'export {var_name}="{escaped_value}"')
        
        lines.append('')
        lines.append('# Commands')
        lines.append('')
        
        # Generate commands based on the command type
        if command == 'run-server':
            lines.extend(self._generate_commands_for_scenario('Server'))
        elif command == 'run-offline':
            lines.extend(self._generate_commands_for_scenario('Offline'))
        elif command == 'run-all':
            lines.extend(self._generate_commands_for_scenario('Server'))
            lines.append('')
            lines.extend(self._generate_commands_for_scenario('Offline'))
        elif command == 'run-performance':
            description = f"{self.config['scenario']} Performance"
            tags = f"test_type:performance,scenario:{self.config['scenario']}"
            if self.config['scenario'] == 'Server':
                description = f"{self.config['scenario']} Performance QPS{self.config['server_target_qps']}"
                tags = f"{tags},qps:{self.config['server_target_qps']}"
            cmd = self.build_command(self.config['scenario'], 'performance', 
                                    self.config['perf_dataset'],
                                    description=description, tags=tags)
            lines.append(self._format_command_as_bash(cmd))
        elif command == 'run-accuracy':
            description = f"{self.config['scenario']} Accuracy"
            tags = f"test_type:accuracy,scenario:{self.config['scenario']}"
            if self.config['scenario'] == 'Server':
                description = f"{self.config['scenario']} Accuracy QPS{self.config['server_target_qps']}"
                tags = f"{tags},qps:{self.config['server_target_qps']}"
            cmd = self.build_command(self.config['scenario'], 'accuracy',
                                    self.config['acc_dataset'],
                                    description=description, tags=tags)
            lines.append(self._format_command_as_bash(cmd))
        elif command == 'run-compliance':
            # Extract scenario from command_args if provided
            scenario = self.config['scenario']
            for arg in command_args:
                if arg in ['Server', 'Offline']:
                    scenario = arg
                    break
            
            # If no specific test, run both
            test_names = [arg for arg in command_args if arg.startswith('TEST')]
            if not test_names:
                lines.extend(self._generate_compliance_commands(scenario, 'TEST07'))
                lines.append('')
                lines.extend(self._generate_compliance_commands(scenario, 'TEST09'))
            else:
                # Generate commands for specified test(s)
                for test_name in test_names:
                    lines.extend(self._generate_compliance_commands(scenario, test_name))
                    if test_name != test_names[-1]:
                        lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_commands_for_scenario(self, scenario: str) -> List[str]:
        """Generate all commands for a scenario."""
        lines = []
        
        # Performance
        description = f"{scenario} Performance"
        tags = f"test_type:performance,scenario:{scenario}"
        if scenario == 'Server':
            description = f"{scenario} Performance QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        cmd = self.build_command(scenario, 'performance', self.config['perf_dataset'],
                               description=description, tags=tags)
        lines.append(f'# {scenario} Performance Test')
        lines.append(self._format_command_as_bash(cmd))
        lines.append('')
        
        # Accuracy
        description = f"{scenario} Accuracy"
        tags = f"test_type:accuracy,scenario:{scenario}"
        if scenario == 'Server':
            description = f"{scenario} Accuracy QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        cmd = self.build_command(scenario, 'accuracy', self.config['acc_dataset'],
                               description=description, tags=tags)
        lines.append(f'# {scenario} Accuracy Test')
        lines.append(self._format_command_as_bash(cmd))
        lines.append('')
        
        # Compliance TEST07
        lines.extend(self._generate_compliance_commands(scenario, 'TEST07'))
        lines.append('')
        
        # Compliance TEST09
        lines.extend(self._generate_compliance_commands(scenario, 'TEST09'))
        
        return lines
    
    def _generate_compliance_commands(self, scenario: str, test_name: str) -> List[str]:
        """Generate compliance test commands."""
        lines = []
        audit_config_path = self.find_audit_config(test_name)
        
        if test_name == 'TEST07':
            dataset_path = self.config['compliance_dataset']
        elif test_name == 'TEST09':
            dataset_path = self.config['perf_dataset']
        else:
            dataset_path = self.config['compliance_dataset']
        
        description = f"{scenario} Compliance {test_name}"
        tags = f"test_type:compliance,scenario:{scenario},compliance_test:{test_name}"
        
        if scenario == 'Server':
            description = f"{scenario} Compliance {test_name} QPS{self.config['server_target_qps']}"
            tags = f"{tags},qps:{self.config['server_target_qps']}"
        
        cmd = self.build_command(scenario, 'performance', dataset_path, test_name.lower(),
                               description, tags, audit_config_path)
        
        lines.append(f'# {scenario} Compliance {test_name}')
        if audit_config_path:
            lines.append(f'# Copy audit.config: {audit_config_path} -> {self.harness_dir / "audit.config"}')
            lines.append(f'cp "{audit_config_path}" "{self.harness_dir / "audit.config"}"')
            lines.append('')
        lines.append(self._format_command_as_bash(cmd))
        if audit_config_path:
            lines.append('')
            lines.append(f'# Cleanup audit.config')
            lines.append(f'rm -f "{self.harness_dir / "audit.config"}"')
        
        return lines
    
    def _format_command_as_bash(self, cmd: List[str]) -> str:
        """Format command as a bash command line."""
        # Quote arguments that need quoting
        quoted_cmd = []
        prev_arg = None
        for i, arg in enumerate(cmd):
            # Always quote --mlflow-tag values and --accuracy-script values
            if prev_arg in ['--mlflow-tag', '--accuracy-script']:
                quoted_cmd.append(f'"{arg}"')
            elif ' ' in arg or '/' in arg or '$' in arg or ':' in arg:
                quoted_cmd.append(f'"{arg}"')
            else:
                quoted_cmd.append(arg)
            prev_arg = arg
        
        # Format as multi-line with continuation
        result = ' '.join(quoted_cmd)
        # Split on ' --' to put each argument on a new line
        formatted = result.replace(' --', ' \\\n    --')
        return formatted
    
    def run_all_tests(self, scenario: str) -> bool:
        """Run all tests for a scenario."""
        print("==========================================")
        print(f"Running all tests for {scenario} scenario")
        print("==========================================")
        print()
        
        # Run performance
        print(">>> Running performance test...")
        if not self.run_performance(scenario):
            return False
        print()
        
        # Run accuracy
        print(">>> Running accuracy test...")
        if not self.run_accuracy(scenario):
            return False
        print()
        
        # Run compliance tests (both TEST07 and TEST09)
        print(">>> Running compliance test TEST07...")
        if not self.run_compliance(scenario, 'TEST07'):
            return False
        print()
        
        print(">>> Running compliance test TEST09...")
        if not self.run_compliance(scenario, 'TEST09'):
            return False
        print()
        
        print(f"✓ All {scenario} tests completed successfully")
        return True
    
    def run(self, args: List[str]):
        """Main run method."""
        # Clean up audit.config at the beginning (both for --print-bash and normal execution)
        audit_dest = self.harness_dir / 'audit.config'
        if audit_dest.exists():
            audit_dest.unlink()
            print("Cleaned up audit.config from harness directory")
        
        # Log dry-run mode
        if '--dry-run' in args:
            print("==========================================")
            print("[DRY RUN MODE ENABLED]")
            print("Commands will be displayed but NOT executed")
            print("==========================================")
            print()
        
        # Parse arguments
        command, command_args = self.parse_args(args)
        
        # Handle print-bash mode
        if self.config['print_bash']:
            # Validate required environment variables before generating bash script
            required_vars = {
                'dataset_dir': 'DATASET_DIR',
                'api_server_url': 'API_SERVER_URL',
                'aws_access_key_id': 'AWS_ACCESS_KEY_ID',
                'aws_secret_access_key': 'AWS_SECRET_ACCESS_KEY',
            }
            
            # Add MLflow requirements only if not disabled
            if not self.config['no_mlflow']:
                required_vars['mlflow_tracking_uri'] = 'MLFLOW_TRACKING_URI'
                required_vars['mlflow_experiment_name'] = 'MLFLOW_EXPERIMENT_NAME'
            
            missing_vars = []
            for config_key, env_var in required_vars.items():
                if not self.config[config_key]:
                    missing_vars.append(env_var)
            
            if missing_vars:
                print("ERROR: The following required environment variables are not set:")
                for var in missing_vars:
                    print(f"  - {var}")
                print("\nPlease set them before using --print-bash")
                sys.exit(1)
            
            # Check server-target-qps for Server scenario
            if self.config['scenario'] == 'Server':
                if not self.config['server_target_qps_set'] or not self.config['server_target_qps']:
                    print("ERROR: --server-target-qps is required for Server scenario")
                    print("       Please specify it via: --server-target-qps <value>")
                    print("       Or set it via environment variable: export SERVER_TARGET_QPS=<value>")
                    sys.exit(1)
            
            bash_script = self.generate_bash_script(command, command_args)
            print(bash_script)
            sys.exit(0)
        
        # Print configuration
        print("==========================================")
        print("Configuration:")
        print(f"  DATASET_DIR: {self.config['dataset_dir']}")
        print(f"  PERF_DATASET: {self.config['perf_dataset']}")
        print(f"  ACC_DATASET: {self.config['acc_dataset']}")
        print(f"  COMPLIANCE_DATASET: {self.config['compliance_dataset']}")
        print(f"  OUTPUT_DIR: {self.config['output_dir']}")
        print(f"  API_SERVER_URL: {self.config['api_server_url']}")
        print(f"  SCENARIO: {self.config['scenario']}")
        if self.config['scenario'] == 'Server':
            print(f"  SERVER_TARGET_QPS: {self.config['server_target_qps']}")
        if not self.config['no_mlflow']:
            print(f"  MLFLOW_EXPERIMENT_NAME: {self.config['mlflow_experiment_name']}")
            if self.config['mlflow_tracking_uri']:
                print(f"  MLFLOW_TRACKING_URI: {self.config['mlflow_tracking_uri']}")
        else:
            print(f"  MLFLOW: Disabled (--no-mlflow)")
        if self.config['aws_access_key_id']:
            print(f"  AWS_ACCESS_KEY_ID: {self.config['aws_access_key_id'][:4]}...")
        if self.config['hf_home']:
            print(f"  HF_HOME: {self.config['hf_home']}")
        if self.config['dry_run']:
            print("  DRY_RUN: true")
        if self.config['user_conf']:
            print(f"  USER_CONF: {self.config['user_conf']}")
        print("==========================================")
        print()
        
        # Validate configuration
        if not self.validate_config():
            print("ERROR: Configuration validation failed. Exiting.")
            sys.exit(1)
        
        # Set ulimit for file descriptors
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (32768, 32768))
            print(f"Set ulimit -n to 32768")
        except Exception as e:
            print(f"WARNING: Could not set ulimit -n 32768: {e}")
            print("         You may need to set it manually: ulimit -n 32768")
        
        # Export environment variables
        if self.config['aws_access_key_id']:
            os.environ['AWS_ACCESS_KEY_ID'] = self.config['aws_access_key_id']
        if self.config['aws_secret_access_key']:
            os.environ['AWS_SECRET_ACCESS_KEY'] = self.config['aws_secret_access_key']
        if self.config['hf_home']:
            os.environ['HF_HOME'] = self.config['hf_home']
        
        # Execute command
        success = True
        
        if command == 'run-server':
            success = self.run_all_tests('Server')
        elif command == 'run-offline':
            success = self.run_all_tests('Offline')
        elif command == 'run-all':
            print("==========================================")
            print("Running all tests for both Server and Offline scenarios")
            print("==========================================")
            print()
            
            print(">>> Running all Server tests...")
            if not self.run_all_tests('Server'):
                success = False
            print()
            
            print(">>> Running all Offline tests...")
            if not self.run_all_tests('Offline'):
                success = False
            print()
            
            if success:
                print("✓ All tests for both scenarios completed successfully")
        elif command == 'run-performance':
            success = self.run_performance(self.config['scenario'])
        elif command == 'run-accuracy':
            success = self.run_accuracy(self.config['scenario'])
        elif command == 'run-compliance':
            # Extract scenario and test from command_args if provided
            compliance_test = None
            for arg in command_args:
                if arg in ['Server', 'Offline']:
                    self.config['scenario'] = arg
                elif arg.startswith('TEST'):
                    compliance_test = arg
            # If no test specified, run_compliance will run both TEST07 and TEST09
            success = self.run_compliance(self.config['scenario'], compliance_test)
        
        sys.exit(0 if success else 1)


def main():
    """Main entry point."""
    runner = HarnessRunner()
    runner.run(sys.argv[1:])


if __name__ == '__main__':
    main()
