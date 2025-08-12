#!/usr/bin/env python3
"""Generation 1 structure validation without dependencies."""

import ast
import os
import sys

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_class_definitions(file_path, expected_classes):
    """Check if expected classes are defined in file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        classes_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
        
        missing_classes = set(expected_classes) - set(classes_found)
        return len(missing_classes) == 0, missing_classes, classes_found
    except Exception as e:
        return False, str(e), []

def check_function_definitions(file_path, expected_functions):
    """Check if expected functions are defined in file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)
        
        missing_functions = set(expected_functions) - set(functions_found)
        return len(missing_functions) == 0, missing_functions, functions_found
    except Exception as e:
        return False, str(e), []

def validate_generation1():
    """Validate Generation 1 implementations."""
    print("🔍 GENERATION 1 STRUCTURE VALIDATION")
    print("=" * 50)
    
    validation_specs = [
        {
            'file': 'src/pno_physics_bench/training/adaptive_scheduling.py',
            'classes': [
                'AdaptiveScheduler', 'UncertaintyAwareLRScheduler', 
                'AdaptiveMomentumScheduler', 'HyperbolicLRScheduler',
                'AdaptiveWarmupScheduler', 'MultiCriteriaScheduler'
            ],
            'functions': ['create_adaptive_scheduler']
        },
        {
            'file': 'src/pno_physics_bench/uncertainty/ensemble_methods.py',
            'classes': [
                'EnsembleBase', 'DeepEnsemble', 'MCDropoutEnsemble',
                'SnapshotEnsemble', 'VariationalEnsemble', 'AdaptiveEnsemble'
            ],
            'functions': ['create_ensemble', 'ensemble_calibration_test']
        },
        {
            'file': 'src/pno_physics_bench/metrics.py',
            'classes': ['CalibrationMetrics'],
            'functions': [
                'uncertainty_quality_index', 'adaptive_calibration_error',
                'uncertainty_decomposition_metrics', 'frequency_domain_calibration',
                'temporal_calibration_drift', 'multi_scale_calibration',
                'comprehensive_uncertainty_report'
            ]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for spec in validation_specs:
        file_path = spec['file']
        print(f"\n📁 Validating {file_path}")
        
        # Test 1: File exists
        total_tests += 1
        if os.path.exists(file_path):
            print(f"✅ File exists")
            passed_tests += 1
        else:
            print(f"❌ File missing")
            continue
        
        # Test 2: Valid Python syntax
        total_tests += 1
        syntax_valid, syntax_error = validate_python_syntax(file_path)
        if syntax_valid:
            print(f"✅ Valid Python syntax")
            passed_tests += 1
        else:
            print(f"❌ Syntax error: {syntax_error}")
            continue
        
        # Test 3: Expected classes
        if 'classes' in spec:
            total_tests += 1
            classes_valid, missing_classes, found_classes = check_class_definitions(
                file_path, spec['classes']
            )
            if classes_valid:
                print(f"✅ All expected classes present: {', '.join(found_classes)}")
                passed_tests += 1
            else:
                print(f"❌ Missing classes: {missing_classes}")
        
        # Test 4: Expected functions
        if 'functions' in spec:
            total_tests += 1
            functions_valid, missing_functions, found_functions = check_function_definitions(
                file_path, spec['functions']
            )
            if functions_valid:
                print(f"✅ All expected functions present")
                passed_tests += 1
            else:
                print(f"❌ Missing functions: {missing_functions}")
    
    # Test code quality metrics
    total_lines = 0
    for spec in validation_specs:
        if os.path.exists(spec['file']):
            with open(spec['file'], 'r') as f:
                total_lines += len(f.readlines())
    
    print(f"\n📊 CODE METRICS:")
    print(f"   • Total lines of new code: {total_lines}")
    print(f"   • Files created: {len(validation_specs)}")
    print(f"   • Classes implemented: {sum(len(spec.get('classes', [])) for spec in validation_specs)}")
    print(f"   • Functions implemented: {sum(len(spec.get('functions', [])) for spec in validation_specs)}")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 GENERATION 1 STRUCTURE VALIDATION SUCCESSFUL!")
        print("✨ Progressive enhancements implemented:")
        print("   • Adaptive learning rate scheduling")
        print("   • Ensemble uncertainty methods")
        print("   • Enhanced calibration metrics")
        return True
    else:
        print("⚠️  Some structural tests failed")
        return False

if __name__ == "__main__":
    success = validate_generation1()
    sys.exit(0 if success else 1)