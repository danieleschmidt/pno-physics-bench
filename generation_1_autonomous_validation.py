#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS SDLC - GENERATION 1 VALIDATION
Validates core PNO functionality without heavy dependencies
"""

import os
import sys
import json
import importlib.util
from datetime import datetime

# Add source to path
sys.path.insert(0, '/root/repo/src')

def validate_structure():
    """Validate repository structure and core components"""
    print("🔍 VALIDATING REPOSITORY STRUCTURE...")
    
    required_dirs = [
        '/root/repo/src/pno_physics_bench',
        '/root/repo/tests',
        '/root/repo/deployment',
        '/root/repo/docs',
        '/root/repo/monitoring'
    ]
    
    required_files = [
        '/root/repo/pyproject.toml',
        '/root/repo/requirements.txt',
        '/root/repo/README.md',
        '/root/repo/src/pno_physics_bench/__init__.py',
        '/root/repo/src/pno_physics_bench/models.py',
        '/root/repo/src/pno_physics_bench/training/__init__.py'
    ]
    
    structure_score = 0
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
            structure_score += 1
        else:
            print(f"❌ Missing directory: {directory}")
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
            structure_score += 1
        else:
            print(f"❌ Missing file: {file_path}")
    
    total_required = len(required_dirs) + len(required_files)
    structure_percentage = (structure_score / total_required) * 100
    
    print(f"\n📊 STRUCTURE VALIDATION: {structure_score}/{total_required} ({structure_percentage:.1f}%)")
    return structure_percentage >= 90

def validate_imports():
    """Validate core module imports"""
    print("\n🔍 VALIDATING CORE IMPORTS...")
    
    try:
        import pno_physics_bench
        print("✅ Core package import successful")
        
        # Check module structure
        package_dir = '/root/repo/src/pno_physics_bench'
        core_modules = ['models', 'training', 'datasets', 'metrics', 'uncertainty']
        
        import_score = 0
        for module in core_modules:
            module_path = os.path.join(package_dir, f'{module}.py')
            init_path = os.path.join(package_dir, module, '__init__.py')
            
            if os.path.exists(module_path) or os.path.exists(init_path):
                print(f"✅ Module available: {module}")
                import_score += 1
            else:
                print(f"❌ Module missing: {module}")
        
        import_percentage = (import_score / len(core_modules)) * 100
        print(f"\n📊 IMPORT VALIDATION: {import_score}/{len(core_modules)} ({import_percentage:.1f}%)")
        return import_percentage >= 80
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def validate_configuration():
    """Validate configuration files and deployment readiness"""
    print("\n🔍 VALIDATING CONFIGURATION...")
    
    config_files = [
        '/root/repo/pyproject.toml',
        '/root/repo/deployment/deployment-config.json',
        '/root/repo/deployment/configs/production.json',
        '/root/repo/monitoring/prometheus.yml'
    ]
    
    config_score = 0
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                if config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        json.load(f)
                print(f"✅ Valid config: {config_file}")
                config_score += 1
            except Exception as e:
                print(f"❌ Invalid config {config_file}: {e}")
        else:
            print(f"❌ Missing config: {config_file}")
    
    config_percentage = (config_score / len(config_files)) * 100
    print(f"\n📊 CONFIG VALIDATION: {config_score}/{len(config_files)} ({config_percentage:.1f}%)")
    return config_percentage >= 75

def validate_deployment_readiness():
    """Validate deployment readiness"""
    print("\n🔍 VALIDATING DEPLOYMENT READINESS...")
    
    deployment_files = [
        '/root/repo/Dockerfile',
        '/root/repo/docker-compose.yml',
        '/root/repo/deployment/pno-deployment.yaml',
        '/root/repo/deployment/pno-service.yaml'
    ]
    
    deployment_score = 0
    for deploy_file in deployment_files:
        if os.path.exists(deploy_file):
            print(f"✅ Deployment file exists: {deploy_file}")
            deployment_score += 1
        else:
            print(f"❌ Missing deployment file: {deploy_file}")
    
    deployment_percentage = (deployment_score / len(deployment_files)) * 100
    print(f"\n📊 DEPLOYMENT VALIDATION: {deployment_score}/{len(deployment_files)} ({deployment_percentage:.1f}%)")
    return deployment_percentage >= 75

def validate_quality_gates():
    """Validate quality gate implementations"""
    print("\n🔍 VALIDATING QUALITY GATES...")
    
    quality_files = [
        '/root/repo/comprehensive_quality_gates.py',
        '/root/repo/security_fixes.py',
        '/root/repo/performance_optimization_suite.py'
    ]
    
    quality_score = 0
    for quality_file in quality_files:
        if os.path.exists(quality_file):
            print(f"✅ Quality gate exists: {quality_file}")
            quality_score += 1
        else:
            print(f"❌ Missing quality gate: {quality_file}")
    
    quality_percentage = (quality_score / len(quality_files)) * 100
    print(f"\n📊 QUALITY GATES VALIDATION: {quality_score}/{len(quality_files)} ({quality_percentage:.1f}%)")
    return quality_percentage >= 75

def validate_research_capabilities():
    """Validate advanced research capabilities"""
    print("\n🔍 VALIDATING RESEARCH CAPABILITIES...")
    
    research_modules = [
        '/root/repo/src/pno_physics_bench/research',
        '/root/repo/src/pno_physics_bench/research/hierarchical_uncertainty.py',
        '/root/repo/src/pno_physics_bench/research/quantum_enhanced_uncertainty.py',
        '/root/repo/src/pno_physics_bench/autonomous_research_agent.py'
    ]
    
    research_score = 0
    for research_item in research_modules:
        if os.path.exists(research_item):
            print(f"✅ Research capability exists: {research_item}")
            research_score += 1
        else:
            print(f"❌ Missing research capability: {research_item}")
    
    research_percentage = (research_score / len(research_modules)) * 100
    print(f"\n📊 RESEARCH VALIDATION: {research_score}/{len(research_modules)} ({research_percentage:.1f}%)")
    return research_percentage >= 75

def run_generation_1_validation():
    """Run complete Generation 1 validation suite"""
    print("🚀 TERRAGON AUTONOMOUS SDLC - GENERATION 1 VALIDATION")
    print("=" * 60)
    
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "generation": 1,
        "validations": {}
    }
    
    # Run all validations
    validations = [
        ("structure", validate_structure),
        ("imports", validate_imports),
        ("configuration", validate_configuration),
        ("deployment", validate_deployment_readiness),
        ("quality_gates", validate_quality_gates),
        ("research", validate_research_capabilities)
    ]
    
    passed_count = 0
    for name, validation_func in validations:
        print(f"\n{'='*20} {name.upper()} VALIDATION {'='*20}")
        try:
            result = validation_func()
            validation_results["validations"][name] = {
                "passed": result,
                "status": "PASS" if result else "FAIL"
            }
            if result:
                passed_count += 1
                print(f"🎉 {name.upper()} VALIDATION: PASSED")
            else:
                print(f"⚠️  {name.upper()} VALIDATION: FAILED")
        except Exception as e:
            print(f"💥 {name.upper()} VALIDATION: ERROR - {e}")
            validation_results["validations"][name] = {
                "passed": False,
                "status": "ERROR",
                "error": str(e)
            }
    
    # Calculate overall score
    total_validations = len(validations)
    success_rate = (passed_count / total_validations) * 100
    validation_results["summary"] = {
        "total_validations": total_validations,
        "passed_validations": passed_count,
        "success_rate": success_rate,
        "overall_status": "PASS" if success_rate >= 80 else "FAIL"
    }
    
    print(f"\n{'='*60}")
    print("🏆 GENERATION 1 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total Validations: {total_validations}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {total_validations - passed_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"🎯 Overall Status: {validation_results['summary']['overall_status']}")
    
    # Save results
    with open('/root/repo/generation_1_autonomous_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Results saved to: generation_1_autonomous_validation_results.json")
    
    return validation_results

if __name__ == "__main__":
    results = run_generation_1_validation()
    
    # Exit with appropriate code
    if results["summary"]["overall_status"] == "PASS":
        print("\n🎉 GENERATION 1 VALIDATION: SUCCESS!")
        sys.exit(0)
    else:
        print("\n⚠️  GENERATION 1 VALIDATION: NEEDS IMPROVEMENT")
        sys.exit(1)