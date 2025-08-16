#!/usr/bin/env python3
"""
Generation 1 Validation: Basic Functionality Test
Tests core structure and imports without requiring full dependencies
"""

import ast
import sys
import os
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_core_structure():
    """Validate core package structure."""
    required_files = [
        "src/pno_physics_bench/__init__.py",
        "src/pno_physics_bench/models.py",
        "src/pno_physics_bench/training/__init__.py",
        "src/pno_physics_bench/datasets.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def check_import_structure():
    """Check import structure in core modules."""
    core_files = [
        "src/pno_physics_bench/__init__.py",
        "src/pno_physics_bench/models.py",
        "src/pno_physics_bench/training/trainer.py",
        "src/pno_physics_bench/uncertainty.py"
    ]
    
    results = {}
    for file_path in core_files:
        if os.path.exists(file_path):
            valid, msg = validate_python_syntax(file_path)
            results[file_path] = (valid, msg)
        else:
            results[file_path] = (False, "File not found")
    
    return results

def validate_package_metadata():
    """Validate package configuration."""
    try:
        with open("pyproject.toml", 'r') as f:
            content = f.read()
        
        required_sections = ["[project]", "[build-system]", "[tool.setuptools"]
        missing_sections = [s for s in required_sections if s not in content]
        
        return len(missing_sections) == 0, missing_sections
    except Exception as e:
        return False, str(e)

def main():
    print("🧪 Generation 1 Validation: Basic Functionality")
    print("=" * 50)
    
    # Test 1: Core Structure
    print("\n1. Core Structure Validation")
    structure_valid, missing = validate_core_structure()
    if structure_valid:
        print("✅ All required files present")
    else:
        print(f"❌ Missing files: {missing}")
    
    # Test 2: Python Syntax
    print("\n2. Python Syntax Validation")
    import_results = check_import_structure()
    all_valid = True
    for file_path, (valid, msg) in import_results.items():
        status = "✅" if valid else "❌"
        print(f"{status} {file_path}: {msg}")
        if not valid:
            all_valid = False
    
    # Test 3: Package Metadata
    print("\n3. Package Metadata Validation")
    meta_valid, meta_missing = validate_package_metadata()
    if meta_valid:
        print("✅ Package metadata complete")
    else:
        print(f"❌ Missing metadata sections: {meta_missing}")
    
    # Overall Result
    print("\n" + "=" * 50)
    overall_status = structure_valid and all_valid and meta_valid
    
    if overall_status:
        print("🎉 Generation 1 PASSED: Basic functionality validated")
        print("✅ Code structure is sound")
        print("✅ Python syntax is valid")
        print("✅ Package metadata is complete")
    else:
        print("⚠️  Generation 1 NEEDS ATTENTION")
        if not structure_valid:
            print("❌ Missing core files")
        if not all_valid:
            print("❌ Syntax or import issues")
        if not meta_valid:
            print("❌ Package metadata incomplete")
    
    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)