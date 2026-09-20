#!/bin/bash

# Script to disable MD036 and MD031 warnings in markdownlint
# This script provides a quick way to disable both rules

echo "🔧 Disabling MD036 and MD031 markdownlint warnings..."
echo "Running Python script to update .markdownlint.jsonc configuration..."
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python script
python3 "$SCRIPT_DIR/disable_md036_warning.py"

# Check if the Python script succeeded
if [ $? -eq 0 ]; then
    echo
    echo "🎉 MD036 and MD031 warnings have been successfully disabled!"
    echo "Your markdown linter should no longer flag:"
    echo "  - Emphasis text as potential headings (MD036)"
    echo "  - Missing blank lines around code blocks (MD031)"
else
    echo
    echo "❌ Failed to disable MD036 and MD031 warnings."
    echo "Please check the error messages above."
    exit 1
fi 