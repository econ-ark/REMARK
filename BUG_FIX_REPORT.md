# REMARK CLI Bug Fix Report

## Bug Description

**Issue**: `AttributeError: 'NoneType' object has no attribute 'group'` on line 231 of `cli.py`

**Command that triggered**: `python3 cli.py lint REMARKs/HAFiscal.yml`

**Root Cause**: The regex pattern used to extract the file structure from STANDARD.md was not matching the actual format of the code block.

## Technical Details

### Original Code (Buggy)
```python
# Line 228-231 in cli.py
standard = re.search(
    f'```\n\..*?```',  # ❌ Doesn't match ```bash\n.
    f.read(),
    flags=re.I | re.DOTALL
).group(0).strip('`').strip()  # ❌ Crashes if re.search returns None
```

### Problem

The code block in STANDARD.md is formatted as:

````markdown
```bash
.
|-- reproduce.sh
...
```
````

But the regex pattern `f'```\n\.'` expects:
````markdown
```
.
|-- reproduce.sh
...
```
````

This caused `re.search()` to return `None`, leading to an `AttributeError` when calling `.group(0)` on `None`.

## Solution

### Fixed Code
```python
# Lines 228-236 in cli.py (fixed)
standard = re.search(
    f'```[^\n]*\n\..*?```',  # ✅ Matches ```bash\n. or ```\n.
    f.read(),
    flags=re.I | re.DOTALL
)
if standard is None:  # ✅ Graceful error handling
    print('ERROR: Could not find file structure in STANDARD.md')
    print('Expected code block starting with a dot (.) for file tree')
    import sys
    sys.exit(1)
standard = standard.group(0).strip('`').strip()
```

### Changes Made

1. **Updated Regex Pattern**:
   - Old: `f'```\n\.'`
   - New: `f'```[^\n]*\n\.'`
   - The `[^\n]*` allows for optional language identifiers (like `bash`, `python`, etc.) after the opening backticks

2. **Added Error Handling**:
   - Check if `re.search()` returns `None` before calling `.group(0)`
   - Provide helpful error message if pattern doesn't match
   - Exit gracefully with sys.exit(1)

## Testing

### Before Fix
```bash
$ python3 cli.py lint REMARKs/HAFiscal.yml
Traceback (most recent call last):
  File "/private/tmp/REMARK/cli.py", line 231, in <module>
    ).group(0).strip('`').strip()
AttributeError: 'NoneType' object has no attribute 'group'
```

### After Fix
```bash
$ python3 cli.py lint REMARKs/HAFiscal.yml
# (No error - command runs successfully)
```

### Regex Pattern Testing
```python
import re

# Test old pattern
old_pattern = r'```\n\..*?```'
old_result = re.search(old_pattern, content, flags=re.I | re.DOTALL)
# Result: None ❌

# Test new pattern
new_pattern = r'```[^\n]*\n\..*?```'
new_result = re.search(new_pattern, content, flags=re.I | re.DOTALL)
# Result: Match object ✅
```

## Impact

- **Affected Command**: `cli.py lint`
- **Severity**: High (command completely broken)
- **Fix Complexity**: Low (simple regex update + error handling)
- **Breaking Changes**: None
- **Backward Compatibility**: Maintained (new pattern still matches old format)

## Recommendations for REMARK Maintainers

1. **Apply this fix** to the main branch
2. **Add unit tests** for the regex pattern to prevent future regressions
3. **Consider standardizing** code block format in STANDARD.md (either always use language identifier or never use it)
4. **Add integration tests** that run lint on sample repositories

## Files Modified

- `cli.py` (lines 228-236)
  - Updated regex pattern
  - Added None check and error handling

## Verification

The fix has been tested with:
- ✅ HAFiscal-Latest repository (fully REMARK-compliant)
- ✅ Regex pattern matching against actual STANDARD.md content
- ✅ Error handling when pattern doesn't match

## Patch File

```patch
--- cli.py.orig
+++ cli.py
@@ -226,7 +226,13 @@
         to_lint = metadata.keys() if args.all else args.remark
         with open(git_root / 'STANDARD.md') as f:
             standard = re.search(
-                f'```\n\..*?```',
+                f'```[^\n]*\n\..*?```',
                 f.read(),
                 flags=re.I | re.DOTALL
-            ).group(0).strip('`').strip()
+            )
+            if standard is None:
+                print('ERROR: Could not find file structure in STANDARD.md')
+                print('Expected code block starting with a dot (.) for file tree')
+                import sys
+                sys.exit(1)
+            standard = standard.group(0).strip('`').strip()
```

---

**Reported by**: AI Assistant (Cursor/Claude)  
**Date**: December 7, 2025  
**Testing Environment**: Python 3.10.10, macOS  
**Status**: ✅ Fixed and Verified
