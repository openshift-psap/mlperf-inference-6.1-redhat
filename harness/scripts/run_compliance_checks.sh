#!/bin/bash
# Run compliance checks for all compliance tests in the output directory
# Usage: run_compliance_checks.sh <output_dir>

OUTPUT_DIR="$1"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required argument"
    echo "Usage: $0 <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 ./harness_output"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="${SCRIPT_DIR}/check_complaince.sh"

if [ ! -f "$CHECK_SCRIPT" ]; then
    echo "Error: check_complaince.sh not found at $CHECK_SCRIPT"
    exit 1
fi

# Check if DATASET_DIR is set
if [ -z "$DATASET_DIR" ]; then
    echo "Warning: DATASET_DIR environment variable is not set"
    echo "         It may be required for compliance checks"
fi

echo "=========================================="
echo "Running Compliance Checks"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Check script: $CHECK_SCRIPT"
echo "=========================================="
echo ""

# Track overall success
OVERALL_SUCCESS=0
TESTS_FOUND=0

# Find all compliance test directories
# Structure: output_dir/scenario/compliance/test07 or test09
for SCENARIO_DIR in "$OUTPUT_DIR"/*/; do
    if [ ! -d "$SCENARIO_DIR" ]; then
        continue
    fi
    
    SCENARIO=$(basename "$SCENARIO_DIR")
    COMPLIANCE_DIR="${SCENARIO_DIR}compliance"
    
    if [ ! -d "$COMPLIANCE_DIR" ]; then
        continue
    fi
    
    # Check for test07 and test09 directories
    for TEST_DIR in "$COMPLIANCE_DIR"/test*/; do
        if [ ! -d "$TEST_DIR" ]; then
            continue
        fi
        
        TEST_NAME=$(basename "$TEST_DIR")
        # Normalize to uppercase (test07 -> TEST07, test09 -> TEST09)
        TEST_NAME_UPPER=$(echo "$TEST_NAME" | tr '[:lower:]' '[:upper:]')
        
        # Only process test07 and test09
        if [ "$TEST_NAME_UPPER" != "TEST07" ] && [ "$TEST_NAME_UPPER" != "TEST09" ]; then
            continue
        fi
        
        TESTS_FOUND=$((TESTS_FOUND + 1))
        
        echo "=========================================="
        echo "Checking: $SCENARIO / $TEST_NAME_UPPER"
        echo "Test directory: $TEST_DIR"
        echo "=========================================="
        echo ""
        
        # Run the check script
        if bash "$CHECK_SCRIPT" "$TEST_DIR" "$TEST_NAME_UPPER"; then
            echo ""
            echo "✓ Compliance check for $SCENARIO / $TEST_NAME_UPPER completed successfully"
        else
            echo ""
            echo "✗ Compliance check for $SCENARIO / $TEST_NAME_UPPER failed"
            OVERALL_SUCCESS=1
        fi
        echo ""
    done
done

echo "=========================================="
if [ $TESTS_FOUND -eq 0 ]; then
    echo "No compliance tests found in: $OUTPUT_DIR"
    echo "Expected structure: <output_dir>/<scenario>/compliance/test07 or test09"
    exit 1
elif [ $OVERALL_SUCCESS -eq 0 ]; then
    echo "✓ All compliance checks completed successfully"
    echo "  Found and checked $TESTS_FOUND test(s)"
else
    echo "✗ Some compliance checks failed"
    echo "  Found and checked $TESTS_FOUND test(s)"
fi
echo "=========================================="

exit $OVERALL_SUCCESS
