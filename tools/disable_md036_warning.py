#!/usr/bin/env python3
"""
Script to disable MD036, MD031, MD004, and MD025 warnings in markdownlint configuration.

This script modifies the .markdownlint.jsonc file to disable:
- MD036 rule: flags emphasis (bold/italic) text that might be intended as headings
- MD031 rule: requires blank lines around fenced code blocks
- MD004 rule: enforces consistent unordered list style (dash vs asterisk)
- MD025 rule: enforces single top-level heading per document
"""

import json
import re
import os
from pathlib import Path

def disable_markdown_warnings():
    """Disable MD036, MD031, MD004, and MD025 warnings in markdownlint configuration."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    config_file = script_dir / '.markdownlint.jsonc'
    
    if not config_file.exists():
        print(f"Error: Configuration file {config_file} not found!")
        return False
    
    # Read the current configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Reading configuration from: {config_file}")
        
        # Remove comments and parse JSON
        # This is a simple approach - remove lines that start with //
        lines = content.split('\n')
        json_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith('//'):
                json_lines.append(line)
        
        json_content = '\n'.join(json_lines)
        
        # Parse the JSON
        config = json.loads(json_content)
        
        # Check if all rules are already disabled
        md036_disabled = 'MD036' in config and config['MD036'] == False
        md031_disabled = 'MD031' in config and config['MD031'] == False
        md004_disabled = 'MD004' in config and config['MD004'] == False
        md025_disabled = 'MD025' in config and config['MD025'] == False
        
        if md036_disabled and md031_disabled and md004_disabled and md025_disabled:
            print("MD036, MD031, MD004, and MD025 are already disabled in the configuration.")
            return True
        
        # Add disable rules
        config['MD036'] = False
        config['MD031'] = False
        config['MD004'] = False
        config['MD025'] = False
        
        # Convert back to JSON with proper formatting
        json_output = json.dumps(config, indent=2)
        
        # Reconstruct the file with comments
        output_lines = [
            '{',
            '  // See https://github.com/DavidAnson/markdownlint/blob/main/schema/.markdownlint.jsonc for schema information',
            '',
            '  // Default state for all rules',
            '  "default": true,',
            '',
            '  // MD004/ul-style: Disabled to allow mixed unordered list styles (dash and asterisk).',
            '  // This prevents warnings when using both - and * for list items.',
            '  "MD004": false,',
            '',
            '  // MD013/line-length: Disabled because it\'s often impractical and not a major concern for GFM.',
            '  "MD013": false,',
            '',
            '  // MD031/blanks-around-fences: Disabled to allow more flexible code block formatting.',
            '  // This prevents warnings when code blocks don\'t have blank lines above/below.',
            '  "MD031": false,',
            '',
            '  // MD033/no-inline-html: Disabled to allow for MyST roles and directives,',
            '  // which can be misinterpreted as inline HTML by the linter. This is crucial for MyST compatibility.',
            '  "MD033": false,',
            '',
            '  // MD034/no-bare-urls: Disabled as a style preference. Bare URLs are common and often desirable.',
            '  "MD034": false,',
            '',
            '  // MD036/no-emphasis-as-heading: Disabled to allow emphasis that is not intended as headings.',
            '  // This prevents false positives when bold/italic text is used for legitimate emphasis.',
            '  "MD036": false,',
            '',
            '  // MD007/ul-indent: Set indent to 4 spaces to be consistent with many formatters.',
            '  "MD007": {',
            '    "indent": 4',
            '  },',
            '',
            '  // MD024/no-duplicate-heading: Allow duplicate headings within the same document.',
            '  // Useful for structuring complex documents without worrying about repeating section names.',
            '  "MD024": false,',
            '',
            '  // MD025/single-title: Disabled to allow multiple top-level headings in the same document.',
            '  // This is useful for documents with multiple main sections or composite documents.',
            '  "MD025": false',
            '}'
        ]
        
        # Write the updated configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines) + '\n')
        
        print(f"✅ Successfully disabled MD036, MD031, MD004, and MD025 warnings in {config_file}")
        print("The following rules have been disabled:")
        print("  - MD036 (no-emphasis-as-heading): Allows emphasis text without heading warnings")
        print("  - MD031 (blanks-around-fences): Allows flexible code block formatting")
        print("  - MD004 (ul-style): Allows mixed unordered list styles (dash and asterisk)")
        print("  - MD025 (single-title): Allows multiple top-level headings per document")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}")
        return False
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return False

def main():
    """Main function to run the script."""
    print("🔧 Disabling MD036, MD031, MD004, and MD025 markdownlint warnings...")
    print("This will modify .markdownlint.jsonc to disable:")
    print("  - MD036: 'emphasis used instead of heading' warning")
    print("  - MD031: 'blanks around fences' warning")
    print("  - MD004: 'unordered list style' warning")
    print("  - MD025: 'multiple top-level headings' warning")
    print()
    
    success = disable_markdown_warnings()
    
    if success:
        print()
        print("✅ Configuration updated successfully!")
        print("The MD036 and MD031 warnings should no longer appear in your markdown linter.")
        print()
        print("Note: You may need to restart your editor or reload the markdown linter")
        print("for the changes to take effect.")
    else:
        print()
        print("❌ Failed to update configuration.")
        print("Please check the error messages above and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 