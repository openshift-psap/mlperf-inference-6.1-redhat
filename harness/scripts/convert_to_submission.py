#!/usr/bin/env python3
"""
Convert Run Submission Output to MLPerf Submission Structure
===========================================================
Converts the output directory from run_submission.py to the MLPerf submission
directory structure (like sample_1 but with RedHat instead of AMD).

Usage:
    python3 convert_to_submission.py --input-dir <output_dir> --output-dir <submission_dir> \\
        --system-name <system_name> --model <model_name> [--division closed|open]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


class SubmissionConverter:
    """Convert run submission output to MLPerf submission structure."""
    
    def __init__(self, input_dir: str, output_dir: str, system_name: str, 
                 model_name: str, division: str = 'closed', debug: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.system_name = system_name
        self.model_name = model_name
        self.division = division
        self.organization = 'RedHat'
        self.debug = debug
        
        # Find loadgen/mlperf.conf relative to script location
        script_dir = Path(__file__).parent.resolve()
        self.loadgen_mlperf_conf = script_dir / '..' / '..' / 'loadgen' / 'mlperf.conf'
        self.loadgen_mlperf_conf = self.loadgen_mlperf_conf.resolve()
        
        # Find system JSON file and config files relative to script location
        self.system_json_src = script_dir / '8xH200-LLM-D-Openshift.json'
        self.harness_dir = script_dir.parent
        self.offline_conf_src = self.harness_dir / 'offline.conf'
        self.default_conf_src = self.harness_dir / 'default.conf'
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def convert(self):
        """Perform the conversion."""
        print("==========================================")
        print("Converting to MLPerf Submission Structure")
        print("==========================================")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"System name: {self.system_name}")
        print(f"Model name: {self.model_name}")
        print(f"Division: {self.division}")
        print(f"Organization: {self.organization}")
        print("==========================================")
        print()
        
        # Create base structure
        base_path = self.output_dir / self.division / self.organization
        results_path = base_path / 'results' / self.system_name / self.model_name
        src_path = base_path / 'src' / self.model_name
        systems_path = base_path / 'systems'
        docs_path = base_path / 'documentation'
        
        # Create directories
        results_path.mkdir(parents=True, exist_ok=True)
        src_path.mkdir(parents=True, exist_ok=True)
        systems_path.mkdir(parents=True, exist_ok=True)
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Copy system JSON file to systems directory
        if self.system_json_src.exists():
            system_json_dest = systems_path / self.system_json_src.name
            shutil.copy2(self.system_json_src, system_json_dest)
            if self.debug:
                print(f"  [DEBUG] Copied: {self.system_json_src} -> {system_json_dest}")
            print(f"  Copied system JSON file to systems directory: {system_json_dest.name}")
        else:
            print(f"  Warning: System JSON file not found at {self.system_json_src}")
        
        # Convert scenarios
        for scenario_dir in self.input_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            scenario_name = scenario_dir.name.capitalize()  # server -> Server, offline -> Offline
            if scenario_name not in ['Server', 'Offline', 'Interactive', 'SingleStream']:
                print(f"Skipping unknown scenario: {scenario_name}")
                continue
            
            print(f"Converting {scenario_name} scenario...")
            self._convert_scenario(scenario_dir, results_path / scenario_name)
        
        # Copy harness_main.py to src subdirectory
        harness_main_src = self.harness_dir / 'harness_main.py'
        if harness_main_src.exists():
            harness_main_dest = src_path / 'harness_main.py'
            shutil.copy2(harness_main_src, harness_main_dest)
            if self.debug:
                print(f"  [DEBUG] Copied: {harness_main_src} -> {harness_main_dest}")
            print(f"  Copied harness_main.py to src/{self.model_name}")
        else:
            print(f"  Warning: harness_main.py not found at {harness_main_src}")
        
        # Create placeholder files
        self._create_placeholder_files(src_path, systems_path, docs_path)
        
        print()
        print("✓ Conversion completed successfully!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Results: {results_path}")
        print(f"  Source: {src_path}")
        print(f"  Systems: {systems_path}")
        print(f"  Documentation: {docs_path}")
    
    def _convert_scenario(self, input_scenario_dir: Path, output_scenario_dir: Path):
        """Convert a single scenario directory."""
        output_scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert accuracy directory
        input_accuracy = input_scenario_dir / 'accuracy'
        if input_accuracy.exists():
            output_accuracy = output_scenario_dir / 'accuracy'
            output_accuracy.mkdir(parents=True, exist_ok=True)
            
            # Copy files from mlperf subdirectory directly to accuracy output (no run_1 subdirectory)
            input_mlperf = input_accuracy / 'mlperf'
            if input_mlperf.exists():
                # Copy only files from mlperf subdirectory (no subdirectories)
                for item in input_mlperf.iterdir():
                    if item.is_file():
                        dest = output_accuracy / item.name
                        shutil.copy2(item, dest)
                        if self.debug:
                            print(f"    [DEBUG] Copied: {item} -> {dest}")
                print(f"  Copied accuracy data")
            else:
                print(f"  Warning: mlperf subdirectory not found in {input_accuracy}")
            
            # Copy accuracy.txt if it exists
            input_accuracy_txt = input_accuracy / 'accuracy.txt'
            if input_accuracy_txt.exists():
                output_accuracy_txt = output_accuracy / 'accuracy.txt'
                shutil.copy2(input_accuracy_txt, output_accuracy_txt)
                if self.debug:
                    print(f"    [DEBUG] Copied: {input_accuracy_txt} -> {output_accuracy_txt}")
                print(f"  Copied accuracy.txt to accuracy subdirectory")
        
        # Convert performance directory
        input_performance = input_scenario_dir / 'performance'
        if input_performance.exists():
            # Create run_1 subdirectory and copy files from mlperf subdirectory
            input_mlperf = input_performance / 'mlperf'
            if input_mlperf.exists():
                output_performance = output_scenario_dir / 'performance' / 'run_1'
                output_performance.mkdir(parents=True, exist_ok=True)
                # Copy only files from mlperf subdirectory to run_1 (no subdirectories)
                for item in input_mlperf.iterdir():
                    if item.is_file():
                        dest = output_performance / item.name
                        shutil.copy2(item, dest)
                        if self.debug:
                            print(f"    [DEBUG] Copied: {item} -> {dest}")
                print(f"  Copied performance data to run_1")
            else:
                print(f"  Warning: mlperf subdirectory not found in {input_performance}")
        
        # Convert compliance directory
        input_compliance = input_scenario_dir / 'compliance'
        if input_compliance.exists():
            # Copy TEST07 and TEST09 subdirectories from within test07/test09 folders
            for test_dir in input_compliance.iterdir():
                if test_dir.is_dir():
                    # Look for TEST07 or TEST09 subdirectory inside test07/test09
                    test_name_lower = test_dir.name.lower()
                    if test_name_lower in ['test07', 'test09']:
                        test_name = test_name_lower.upper()  # test07 -> TEST07
                        inner_test_dir = test_dir / test_name
                        if inner_test_dir.exists() and inner_test_dir.is_dir():
                            output_test_dir = output_scenario_dir / test_name
                            # Copy the TEST07 or TEST09 subdirectory as-is
                            self._copy_directory(inner_test_dir, output_test_dir)
                            if self.debug:
                                print(f"    [DEBUG] Copied directory: {inner_test_dir} -> {output_test_dir}")
                            print(f"  Copied compliance {test_name} (as-is)")
                        else:
                            print(f"  Warning: {test_name} subdirectory not found in {test_dir}")
        
        # Create required files in scenario output directory
        self._create_scenario_files(output_scenario_dir, output_scenario_dir.name)
    
    def _convert_compliance_test(self, input_test_dir: Path, output_test_dir: Path, test_name: str):
        """Convert a compliance test directory."""
        output_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mlperf directory if it exists
        input_mlperf = input_test_dir / 'mlperf'
        if input_mlperf.exists():
            output_mlperf = output_test_dir / 'mlperf'
            self._copy_directory(input_mlperf, output_mlperf)
            if self.debug:
                print(f"    [DEBUG] Copied directory: {input_mlperf} -> {output_mlperf}")
        
        # Look for accuracy directory in mlperf
        input_accuracy = input_test_dir / 'mlperf' / 'mlperf_log_accuracy.json'
        if input_accuracy.exists():
            output_accuracy_dir = output_test_dir / 'accuracy'
            output_accuracy_dir.mkdir(parents=True, exist_ok=True)
            dest = output_accuracy_dir / 'mlperf_log_accuracy.json'
            shutil.copy2(input_accuracy, dest)
            if self.debug:
                print(f"    [DEBUG] Copied: {input_accuracy} -> {dest}")
        
        # Copy test-specific verify files
        if test_name == 'TEST07':
            verify_file = input_test_dir / 'verify_accuracy.txt'
            if verify_file.exists():
                dest = output_test_dir / 'verify_accuracy.txt'
                shutil.copy2(verify_file, dest)
                if self.debug:
                    print(f"    [DEBUG] Copied: {verify_file} -> {dest}")
                print(f"    Copied verify_accuracy.txt")
        elif test_name == 'TEST09':
            verify_file = input_test_dir / 'verify_output_len.txt'
            if verify_file.exists():
                dest = output_test_dir / 'verify_output_len.txt'
                shutil.copy2(verify_file, dest)
                if self.debug:
                    print(f"    [DEBUG] Copied: {verify_file} -> {dest}")
                print(f"    Copied verify_output_len.txt")
    
    def _copy_directory(self, src: Path, dst: Path):
        """Copy a directory recursively."""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    
    def _create_scenario_files(self, output_scenario_dir: Path, scenario_name: str):
        """Create required files in scenario output directory."""
        # Create measurements.json with default content
        measurements_json = output_scenario_dir / 'measurements.json'
        if not measurements_json.exists():
            measurements_data = {
                "input_data_types": "int32",
                "retraining": "No",
                "starting_weights_filename": "Original Huggingface model weights",
                "weight_data_types": "fp8",
                "weight_transformations": "quantization"
            }
            with open(measurements_json, 'w') as f:
                json.dump(measurements_data, f, indent=4)
            if self.debug:
                print(f"    [DEBUG] Created: {measurements_json}")
            print(f"  Created measurements.json")
        
        # Copy mlperf.conf from loadgen directory
        mlperf_conf = output_scenario_dir / 'mlperf.conf'
        if not mlperf_conf.exists():
            if self.loadgen_mlperf_conf.exists():
                shutil.copy2(self.loadgen_mlperf_conf, mlperf_conf)
                if self.debug:
                    print(f"    [DEBUG] Copied: {self.loadgen_mlperf_conf} -> {mlperf_conf}")
                print(f"  Copied mlperf.conf from loadgen")
            else:
                print(f"  Warning: mlperf.conf not found at {self.loadgen_mlperf_conf}")
        
        # Copy scenario-specific user.conf files
        user_conf = output_scenario_dir / 'user.conf'
        if scenario_name == 'Offline':
            # Copy offline.conf to Offline subdirectory as user.conf
            if self.offline_conf_src.exists():
                shutil.copy2(self.offline_conf_src, user_conf)
                if self.debug:
                    print(f"    [DEBUG] Copied: {self.offline_conf_src} -> {user_conf}")
                print(f"  Copied offline.conf as user.conf")
            else:
                print(f"  Warning: offline.conf not found at {self.offline_conf_src}, creating empty user.conf")
                user_conf.touch()
        elif scenario_name == 'Server':
            # Copy default.conf to Server subdirectory as user.conf
            if self.default_conf_src.exists():
                shutil.copy2(self.default_conf_src, user_conf)
                if self.debug:
                    print(f"    [DEBUG] Copied: {self.default_conf_src} -> {user_conf}")
                print(f"  Copied default.conf as user.conf")
            else:
                print(f"  Warning: default.conf not found at {self.default_conf_src}, creating empty user.conf")
                user_conf.touch()
        else:
            # For other scenarios, create empty user.conf
            if not user_conf.exists():
                user_conf.touch()
                if self.debug:
                    print(f"    [DEBUG] Created: {user_conf}")
                print(f"  Created user.conf")
        
        # Create README.md with basic content
        readme_md = output_scenario_dir / 'README.md'
        if not readme_md.exists():
            scenario_name = output_scenario_dir.name
            readme_content = f"""# {scenario_name} Scenario

This directory contains the MLPerf inference results for the {scenario_name} scenario.

## Contents

- `accuracy/`: Accuracy test results
- `performance/`: Performance test results
- `measurements.json`: Measurement configuration
- `mlperf.conf`: MLPerf configuration file
- `user.conf`: User configuration overrides
"""
            readme_md.write_text(readme_content)
            if self.debug:
                print(f"    [DEBUG] Created: {readme_md}")
            print(f"  Created README.md")
    
    def _create_placeholder_files(self, src_path: Path, systems_path: Path, docs_path: Path):
        """Create placeholder files if they don't exist."""
        # Create README.md in src
        src_readme = src_path / 'README.md'
        if not src_readme.exists():
            src_readme.write_text(f"""# {self.model_name}

Model source code and implementation details.

## Overview

This directory contains the source code and implementation details for {self.model_name}.

## Files

- Implementation files
- Configuration files
- Build scripts
""")
            print(f"  Created placeholder: {src_path / 'README.md'}")
        
        # Create system JSON file
        system_json = systems_path / f'{self.system_name}.json'
        if not system_json.exists():
            system_json.write_text(f"""{{
    "accelerator_frequency": "",
    "accelerator_host_interconnect": "",
    "accelerator_interconnect": "",
    "accelerator_memory_capacity": "",
    "accelerator_memory_configuration": "",
    "accelerators_per_node": "",
    "framework": "",
    "host_memory_capacity": "",
    "host_processor_model_name": "",
    "host_processors_per_node": "",
    "host_storage_capacity": "",
    "host_storage_type": "",
    "hw_notes": "",
    "number_of_nodes": 1,
    "number_of_type_nics_installed": "",
    "operating_system": "",
    "other_software_stack": "",
    "sw_notes": "",
    "system_name": "{self.system_name}",
    "system_type": "datacenter"
}}
""")
            print(f"  Created placeholder: {system_json}")
        
        # Create documentation files
        docs_readme = docs_path / 'README.md'
        if not docs_readme.exists():
            docs_readme.write_text(f"""# {self.organization} Submission Documentation

## Overview

This submission contains results for {self.model_name} running on {self.system_name}.

## Contents

- Calibration documentation
- System configuration details
- Performance tuning notes
""")
            print(f"  Created placeholder: {docs_path / 'README.md'}")
        
        # Create calibration.md
        calibration_md = docs_path / 'calibration.md'
        if not calibration_md.exists():
            calibration_md.write_text(f"""# Calibration Documentation

## Calibration Process

Describe the calibration process for {self.model_name}.

## Calibration Dataset

- Dataset details
- Calibration methodology

## Results

- Calibration accuracy
- Performance impact
""")
            print(f"  Created placeholder: {docs_path / 'calibration.md'}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert run_submission.py output to MLPerf submission structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert output directory
  python3 convert_to_submission.py \\
      --input-dir ./harness_output \\
      --output-dir ./submission \\
      --system-name "8xH100_2xEPYC_9654" \\
      --model "gpt-oss-120b"

  # Convert with open division
  python3 convert_to_submission.py \\
      --input-dir ./harness_output \\
      --output-dir ./submission \\
      --system-name "8xH100_2xEPYC_9654" \\
      --model "gpt-oss-120b" \\
      --division open
        """
    )
    
    parser.add_argument('--input-dir', required=True, help='Input directory from run_submission.py')
    parser.add_argument('--output-dir', required=True, help='Output directory for submission structure')
    parser.add_argument('--system-name', required=True, help='System name (e.g., 8xH100_2xEPYC_9654)')
    parser.add_argument('--model', required=True, help='Model name (e.g., gpt-oss-120b)')
    parser.add_argument('--division', choices=['closed', 'open'], default='closed',
                       help='Division (default: closed)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to print detailed file copy information')
    
    args = parser.parse_args()
    
    try:
        converter = SubmissionConverter(
            args.input_dir,
            args.output_dir,
            args.system_name,
            args.model,
            args.division,
            args.debug
        )
        converter.convert()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
