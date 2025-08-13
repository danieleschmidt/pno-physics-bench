#!/usr/bin/env python3
"""Generation 1 validation and testing script for PNO Physics Bench."""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test that all new modules can be imported."""
    try:
        # Test adaptive scheduling imports
        from pno_physics_bench.training.adaptive_scheduling import (
            UncertaintyAwareLRScheduler,
            AdaptiveMomentumScheduler,
            HyperbolicLRScheduler,
            create_adaptive_scheduler
        )
        print("✅ Adaptive scheduling imports successful")
        
        # Test uncertainty ensemble imports
        from pno_physics_bench.uncertainty.ensemble_methods import (
            DeepEnsemble,
            MCDropoutEnsemble,
            VariationalEnsemble,
            create_ensemble
        )
        print("✅ Ensemble methods imports successful")
        
        # Test enhanced metrics imports
        from pno_physics_bench.metrics import CalibrationMetrics
        print("✅ Enhanced metrics imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_adaptive_scheduler_creation():
    """Test adaptive scheduler creation without dependencies."""
    try:
        from pno_physics_bench.training.adaptive_scheduling import create_adaptive_scheduler
        
        # Mock optimizer class
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 0.001}]
        
        optimizer = MockOptimizer()
        
        # Test uncertainty-aware scheduler creation
        scheduler = create_adaptive_scheduler(
            optimizer, 
            scheduler_type="uncertainty_aware",
            patience=10
        )
        print("✅ Uncertainty-aware scheduler creation successful")
        
        # Test hyperbolic scheduler creation
        scheduler = create_adaptive_scheduler(
            optimizer,
            scheduler_type="hyperbolic",
            T_max=100
        )
        print("✅ Hyperbolic scheduler creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Scheduler creation error: {e}")
        return False

def test_ensemble_factory():
    """Test ensemble factory method."""
    try:
        from pno_physics_bench.uncertainty.ensemble_methods import create_ensemble
        
        # Mock model class
        class MockModel:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
        
        models = [MockModel() for _ in range(3)]
        
        # Test deep ensemble creation
        ensemble = create_ensemble("deep", models)
        print("✅ Deep ensemble creation successful")
        
        # Test adaptive ensemble creation  
        ensemble = create_ensemble("adaptive", models)
        print("✅ Adaptive ensemble creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Ensemble creation error: {e}")
        return False

def test_metrics_functionality():
    """Test enhanced metrics functionality."""
    try:
        from pno_physics_bench.metrics import CalibrationMetrics
        
        # Create metrics instance
        metrics = CalibrationMetrics()
        print("✅ CalibrationMetrics instantiation successful")
        
        # Test method existence
        methods_to_test = [
            'uncertainty_quality_index',
            'adaptive_calibration_error', 
            'uncertainty_decomposition_metrics',
            'comprehensive_uncertainty_report'
        ]
        
        for method in methods_to_test:
            if hasattr(metrics, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Metrics functionality error: {e}")
        return False

def test_code_structure():
    """Test code structure and organization."""
    required_files = [
        'src/pno_physics_bench/training/adaptive_scheduling.py',
        'src/pno_physics_bench/uncertainty/ensemble_methods.py',
        'src/pno_physics_bench/uncertainty/__init__.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            return False
    
    return True

def run_generation1_validation():
    """Run complete Generation 1 validation."""
    print("🧪 GENERATION 1 VALIDATION SUITE")
    print("=" * 50)
    
    tests = [
        ("Code Structure", test_code_structure),
        ("Module Imports", test_imports),
        ("Adaptive Schedulers", test_adaptive_scheduler_creation),
        ("Ensemble Factory", test_ensemble_factory),
        ("Enhanced Metrics", test_metrics_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 1 VALIDATION SUCCESSFUL!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False

if __name__ == "__main__":
    success = run_generation1_validation()
    sys.exit(0 if success else 1)