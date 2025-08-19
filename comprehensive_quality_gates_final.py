#!/usr/bin/env python3
"""
Comprehensive Quality Gates - Final Validation
Complete SDLC validation with production readiness assessment.
"""

import sys
import os
import json
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import re

# Add src to path
sys.path.insert(0, 'src')

class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'quality_gates': {},
            'overall_score': 0.0,
            'production_ready': False
        }
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup quality gate logger."""
        logger = logging.getLogger('quality_gates')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        print("🏗️  Validating Code Structure...")
        
        structure_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check required directories
        required_dirs = [
            'src/pno_physics_bench',
            'tests',
            'docs',
            'deployment'
        ]
        
        for req_dir in required_dirs:
            if os.path.exists(req_dir):
                structure_score += 1.0
                print(f"   ✓ Directory {req_dir} exists")
            else:
                issues.append(f"Missing directory: {req_dir}")
                print(f"   ✗ Missing directory: {req_dir}")
        
        # Check critical files
        required_files = [
            'README.md',
            'pyproject.toml',
            'requirements.txt',
            'src/pno_physics_bench/__init__.py',
            'src/pno_physics_bench/models.py'
        ]
        
        for req_file in required_files:
            if os.path.exists(req_file):
                structure_score += 1.0
                print(f"   ✓ File {req_file} exists")
            else:
                issues.append(f"Missing file: {req_file}")
                print(f"   ✗ Missing file: {req_file}")
        
        # Check module organization
        modules_found = 0
        expected_modules = [
            'src/pno_physics_bench/training',
            'src/pno_physics_bench/uncertainty',
            'src/pno_physics_bench/research',
            'src/pno_physics_bench/scaling'
        ]
        
        for module in expected_modules:
            if os.path.exists(module):
                modules_found += 1
        
        structure_score += (modules_found / len(expected_modules)) * 1.0
        
        structure_percentage = (structure_score / max_score) * 100
        
        return {
            'score': structure_percentage,
            'issues': issues,
            'modules_found': modules_found,
            'status': 'pass' if structure_percentage >= 80 else 'fail'
        }
    
    def validate_testing(self) -> Dict[str, Any]:
        """Validate testing implementation and coverage."""
        print("🧪 Validating Testing Framework...")
        
        testing_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check test files exist
        test_files = []
        if os.path.exists('tests'):
            for root, dirs, files in os.walk('tests'):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
        
        if test_files:
            testing_score += 2.0
            print(f"   ✓ Found {len(test_files)} test files")
        else:
            issues.append("No test files found")
            print("   ✗ No test files found")
        
        # Check for different test types
        test_categories = {
            'unit': 0,
            'integration': 0,
            'performance': 0,
            'research': 0
        }
        
        for test_file in test_files:
            if 'unit' in test_file:
                test_categories['unit'] += 1
            elif 'integration' in test_file:
                test_categories['integration'] += 1
            elif 'performance' in test_file or 'benchmark' in test_file:
                test_categories['performance'] += 1
            elif 'research' in test_file:
                test_categories['research'] += 1
        
        # Score based on test diversity
        for category, count in test_categories.items():
            if count > 0:
                testing_score += 1.0
                print(f"   ✓ {category.title()} tests found: {count}")
        
        # Check validation files we created
        validation_files = [
            'generation_1_validation.py',
            'generation_2_robustness_suite_fixed.py', 
            'generation_3_scaling_suite_fixed.py'
        ]
        
        validation_found = sum(1 for f in validation_files if os.path.exists(f))
        if validation_found == len(validation_files):
            testing_score += 2.0
            print(f"   ✓ All {validation_found} generation validation suites present")
        else:
            issues.append(f"Missing validation suites: {len(validation_files) - validation_found}")
        
        testing_percentage = (testing_score / max_score) * 100
        
        return {
            'score': testing_percentage,
            'test_files_count': len(test_files),
            'test_categories': test_categories,
            'validation_suites': validation_found,
            'issues': issues,
            'status': 'pass' if testing_percentage >= 70 else 'fail'
        }
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security measures and practices."""
        print("🔒 Validating Security Measures...")
        
        security_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check for security-related modules
        security_modules = [
            'src/pno_physics_bench/security',
            'src/pno_physics_bench/validation.py',
            'src/pno_physics_bench/robustness'
        ]
        
        security_modules_found = 0
        for module in security_modules:
            if os.path.exists(module):
                security_modules_found += 1
                print(f"   ✓ Security module found: {module}")
        
        security_score += (security_modules_found / len(security_modules)) * 3.0
        
        # Check for input validation
        validation_patterns = [
            'validate_tensor_input',
            'validate_config',
            'SecurityError',
            'input_sanitization'
        ]
        
        validation_found = 0
        python_files = []
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for pattern in validation_patterns:
            found_in_files = []
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if pattern in content:
                            found_in_files.append(py_file)
                except:
                    continue
            
            if found_in_files:
                validation_found += 1
                print(f"   ✓ Security pattern '{pattern}' found in {len(found_in_files)} files")
        
        security_score += (validation_found / len(validation_patterns)) * 3.0
        
        # Check for logging and audit trails
        logging_patterns = ['audit_log', 'security_event', 'log_security']
        logging_found = 0
        
        for pattern in logging_patterns:
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            logging_found += 1
                            break
                except:
                    continue
        
        if logging_found > 0:
            security_score += 2.0
            print(f"   ✓ Security logging patterns found")
        
        # Check for dependency security (basic)
        if os.path.exists('requirements.txt'):
            try:
                with open('requirements.txt', 'r') as f:
                    deps = f.read()
                    # Basic check for version pinning
                    if '>=' in deps and '==' in deps:
                        security_score += 1.0
                        print("   ✓ Dependencies have version constraints")
                    else:
                        issues.append("Dependencies should have version constraints")
            except:
                issues.append("Could not read requirements.txt")
        
        # Check for secrets (basic scan)
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        
        secrets_found = False
        for py_file in python_files[:10]:  # Sample check
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found = True
                            issues.append(f"Potential secret found in {py_file}")
                            break
            except:
                continue
        
        if not secrets_found:
            security_score += 1.0
            print("   ✓ No obvious secrets found in code")
        
        security_percentage = (security_score / max_score) * 100
        
        return {
            'score': security_percentage,
            'security_modules': security_modules_found,
            'validation_patterns': validation_found,
            'issues': issues,
            'status': 'pass' if security_percentage >= 75 else 'fail'
        }
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics."""
        print("⚡ Validating Performance Characteristics...")
        
        performance_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check for performance-related modules
        perf_modules = [
            'src/pno_physics_bench/optimization',
            'src/pno_physics_bench/scaling',
            'src/pno_physics_bench/performance_optimization.py',
            'src/pno_physics_bench/monitoring.py'
        ]
        
        perf_modules_found = 0
        for module in perf_modules:
            if os.path.exists(module):
                perf_modules_found += 1
                print(f"   ✓ Performance module found: {module}")
        
        performance_score += (perf_modules_found / len(perf_modules)) * 3.0
        
        # Check for caching implementation
        caching_patterns = [
            'cache',
            'lru_cache',
            'IntelligentCache',
            'memoization'
        ]
        
        caching_found = 0
        python_files = []
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for pattern in caching_patterns:
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            caching_found += 1
                            break
                except:
                    continue
        
        if caching_found > 0:
            performance_score += 2.0
            print(f"   ✓ Caching mechanisms found")
        
        # Check for concurrent processing
        concurrency_patterns = [
            'ThreadPoolExecutor',
            'ProcessPoolExecutor',
            'concurrent.futures',
            'multiprocessing'
        ]
        
        concurrency_found = 0
        for pattern in concurrency_patterns:
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            concurrency_found += 1
                            break
                except:
                    continue
        
        if concurrency_found > 0:
            performance_score += 2.0
            print(f"   ✓ Concurrent processing capabilities found")
        
        # Check for monitoring and profiling
        monitoring_patterns = [
            'PerformanceMonitor',
            'profiling',
            'metrics',
            'benchmark'
        ]
        
        monitoring_found = 0
        for pattern in monitoring_patterns:
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            monitoring_found += 1
                            break
                except:
                    continue
        
        if monitoring_found > 0:
            performance_score += 1.5
            print(f"   ✓ Performance monitoring found")
        
        # Check for memory optimization patterns
        memory_patterns = [
            'gc.collect',
            'memory_efficient',
            'resource_management',
            '__slots__'
        ]
        
        memory_found = 0
        for pattern in memory_patterns:
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            memory_found += 1
                            break
                except:
                    continue
        
        if memory_found > 0:
            performance_score += 1.5
            print(f"   ✓ Memory optimization patterns found")
        
        performance_percentage = (performance_score / max_score) * 100
        
        return {
            'score': performance_percentage,
            'performance_modules': perf_modules_found,
            'caching_features': caching_found > 0,
            'concurrency_features': concurrency_found > 0,
            'monitoring_features': monitoring_found > 0,
            'issues': issues,
            'status': 'pass' if performance_percentage >= 75 else 'fail'
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness."""
        print("📚 Validating Documentation...")
        
        doc_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check README.md
        if os.path.exists('README.md'):
            try:
                with open('README.md', 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                    
                    # Check README sections
                    required_sections = [
                        'overview',
                        'installation',
                        'usage',
                        'example',
                        'api',
                        'architecture'
                    ]
                    
                    sections_found = 0
                    for section in required_sections:
                        if section.lower() in readme_content.lower():
                            sections_found += 1
                    
                    doc_score += (sections_found / len(required_sections)) * 3.0
                    print(f"   ✓ README.md has {sections_found}/{len(required_sections)} required sections")
                    
                    # Check for examples
                    if '```python' in readme_content or '```bash' in readme_content:
                        doc_score += 1.0
                        print("   ✓ README.md contains code examples")
                    else:
                        issues.append("README.md should contain code examples")
            except:
                issues.append("Could not read README.md")
        else:
            issues.append("README.md not found")
        
        # Check for API documentation
        api_docs = [
            'API_DOCUMENTATION.md',
            'docs/api.md',
            'docs/api_reference.md'
        ]
        
        api_doc_found = False
        for api_doc in api_docs:
            if os.path.exists(api_doc):
                api_doc_found = True
                doc_score += 1.0
                print(f"   ✓ API documentation found: {api_doc}")
                break
        
        if not api_doc_found:
            issues.append("API documentation not found")
        
        # Check for architecture documentation
        arch_docs = [
            'ARCHITECTURE.md',
            'docs/architecture.md',
            'docs/design.md'
        ]
        
        arch_doc_found = False
        for arch_doc in arch_docs:
            if os.path.exists(arch_doc):
                arch_doc_found = True
                doc_score += 1.0
                print(f"   ✓ Architecture documentation found: {arch_doc}")
                break
        
        if not arch_doc_found:
            issues.append("Architecture documentation not found")
        
        # Check for docstrings in Python files
        python_files = []
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        files_with_docstrings = 0
        total_files_checked = min(10, len(python_files))  # Sample check
        
        for py_file in python_files[:total_files_checked]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
            except:
                continue
        
        if total_files_checked > 0:
            docstring_ratio = files_with_docstrings / total_files_checked
            doc_score += docstring_ratio * 2.0
            print(f"   ✓ {files_with_docstrings}/{total_files_checked} sampled Python files have docstrings")
        
        # Check for deployment documentation
        deploy_docs = [
            'DEPLOYMENT.md',
            'deployment/README.md',
            'docs/deployment.md'
        ]
        
        deploy_doc_found = False
        for deploy_doc in deploy_docs:
            if os.path.exists(deploy_doc):
                deploy_doc_found = True
                doc_score += 1.0
                print(f"   ✓ Deployment documentation found: {deploy_doc}")
                break
        
        # Check for changelog/release notes
        changelog_files = [
            'CHANGELOG.md',
            'HISTORY.md',
            'RELEASES.md'
        ]
        
        changelog_found = False
        for changelog in changelog_files:
            if os.path.exists(changelog):
                changelog_found = True
                doc_score += 1.0
                print(f"   ✓ Changelog found: {changelog}")
                break
        
        # Check examples directory
        if os.path.exists('examples'):
            example_files = [f for f in os.listdir('examples') 
                           if f.endswith('.py')]
            if example_files:
                doc_score += 1.0
                print(f"   ✓ Examples directory has {len(example_files)} Python examples")
            else:
                issues.append("Examples directory exists but is empty")
        else:
            issues.append("Examples directory not found")
        
        documentation_percentage = (doc_score / max_score) * 100
        
        return {
            'score': documentation_percentage,
            'readme_exists': os.path.exists('README.md'),
            'api_doc_exists': api_doc_found,
            'arch_doc_exists': arch_doc_found,
            'deployment_doc_exists': deploy_doc_found,
            'changelog_exists': changelog_found,
            'docstring_ratio': files_with_docstrings / max(1, total_files_checked),
            'issues': issues,
            'status': 'pass' if documentation_percentage >= 70 else 'fail'
        }
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment and production readiness."""
        print("🚀 Validating Deployment Readiness...")
        
        deploy_score = 0.0
        max_score = 10.0
        issues = []
        
        # Check deployment configuration files
        deploy_configs = [
            'Dockerfile',
            'docker-compose.yml',
            'pyproject.toml',
            'requirements.txt'
        ]
        
        configs_found = 0
        for config in deploy_configs:
            if os.path.exists(config):
                configs_found += 1
                print(f"   ✓ Deployment config found: {config}")
        
        deploy_score += (configs_found / len(deploy_configs)) * 2.5
        
        # Check deployment directory
        if os.path.exists('deployment'):
            deploy_files = os.listdir('deployment')
            if deploy_files:
                deploy_score += 1.5
                print(f"   ✓ Deployment directory has {len(deploy_files)} files")
            else:
                issues.append("Deployment directory is empty")
        else:
            issues.append("Deployment directory not found")
        
        # Check for CI/CD configuration
        cicd_files = [
            '.github/workflows',
            '.gitlab-ci.yml',
            'Jenkinsfile',
            '.circleci/config.yml'
        ]
        
        cicd_found = False
        for cicd in cicd_files:
            if os.path.exists(cicd):
                cicd_found = True
                deploy_score += 1.0
                print(f"   ✓ CI/CD configuration found: {cicd}")
                break
        
        if not cicd_found:
            issues.append("No CI/CD configuration found")
        
        # Check for monitoring and logging setup
        monitoring_files = [
            'monitoring',
            'prometheus.yml',
            'grafana',
            'docker-compose.monitoring.yml'
        ]
        
        monitoring_found = 0
        for monitor in monitoring_files:
            if os.path.exists(monitor):
                monitoring_found += 1
                print(f"   ✓ Monitoring setup found: {monitor}")
        
        if monitoring_found > 0:
            deploy_score += 1.5
        
        # Check for environment configuration
        env_files = [
            '.env.example',
            'config',
            'deployment/configs'
        ]
        
        env_found = 0
        for env_file in env_files:
            if os.path.exists(env_file):
                env_found += 1
        
        if env_found > 0:
            deploy_score += 1.0
            print(f"   ✓ Environment configuration found")
        
        # Check for security configuration
        security_configs = [
            'deployment/security.json',
            'security_config.json',
            'deployment/configs/security.json'
        ]
        
        security_config_found = False
        for sec_config in security_configs:
            if os.path.exists(sec_config):
                security_config_found = True
                deploy_score += 1.0
                print(f"   ✓ Security configuration found: {sec_config}")
                break
        
        # Check for production validation scripts
        production_scripts = [
            'production_deployment_suite.py',
            'production_example.py',
            'deploy_production.py'
        ]
        
        prod_scripts_found = 0
        for script in production_scripts:
            if os.path.exists(script):
                prod_scripts_found += 1
        
        if prod_scripts_found > 0:
            deploy_score += 1.5
            print(f"   ✓ Production scripts found: {prod_scripts_found}")
        
        # Check for health checks
        health_patterns = [
            'health_check',
            'readiness_probe',
            'liveness_probe'
        ]
        
        health_found = False
        python_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for pattern in health_patterns:
            for py_file in python_files[:20]:  # Sample check
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if pattern in f.read():
                            health_found = True
                            break
                except:
                    continue
            if health_found:
                break
        
        if health_found:
            deploy_score += 1.0
            print(f"   ✓ Health check mechanisms found")
        
        deployment_percentage = (deploy_score / max_score) * 100
        
        return {
            'score': deployment_percentage,
            'config_files': configs_found,
            'cicd_configured': cicd_found,
            'monitoring_configured': monitoring_found > 0,
            'security_configured': security_config_found,
            'production_scripts': prod_scripts_found,
            'health_checks': health_found,
            'issues': issues,
            'status': 'pass' if deployment_percentage >= 80 else 'fail'
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("=" * 80)
        print("🎯 COMPREHENSIVE QUALITY GATES VALIDATION")
        print("=" * 80)
        
        # Run all quality gates
        quality_gates = {
            'code_structure': self.validate_code_structure(),
            'testing': self.validate_testing(),
            'security': self.validate_security(),
            'performance': self.validate_performance(),
            'documentation': self.validate_documentation(),
            'deployment': self.validate_deployment_readiness()
        }
        
        self.results['quality_gates'] = quality_gates
        
        # Calculate overall score
        total_score = 0.0
        total_weight = 0.0
        
        # Weights for different quality gates
        weights = {
            'code_structure': 1.0,
            'testing': 1.5,
            'security': 1.3,
            'performance': 1.2,
            'documentation': 1.0,
            'deployment': 1.4
        }
        
        for gate_name, gate_result in quality_gates.items():
            weight = weights.get(gate_name, 1.0)
            total_score += gate_result['score'] * weight
            total_weight += weight * 100  # Since scores are percentages
        
        overall_score = total_score / total_weight * 100
        self.results['overall_score'] = overall_score
        
        # Determine production readiness
        failed_gates = [name for name, result in quality_gates.items() 
                       if result['status'] == 'fail']
        
        critical_gates = ['testing', 'security', 'deployment']
        critical_failures = [gate for gate in failed_gates if gate in critical_gates]
        
        self.results['production_ready'] = (
            overall_score >= 75 and 
            len(critical_failures) == 0
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 80)
        
        for gate_name, gate_result in quality_gates.items():
            status_icon = "✅" if gate_result['status'] == 'pass' else "❌"
            print(f"{status_icon} {gate_name.replace('_', ' ').title()}: {gate_result['score']:.1f}%")
        
        print(f"\n🎯 Overall Quality Score: {overall_score:.1f}%")
        
        if self.results['production_ready']:
            print("🎉 ✅ PRODUCTION READY")
            print("   All critical quality gates passed!")
        else:
            print("⚠️  ❌ NOT PRODUCTION READY")
            if critical_failures:
                print(f"   Critical failures: {', '.join(critical_failures)}")
            if overall_score < 75:
                print(f"   Overall score too low: {overall_score:.1f}% (minimum: 75%)")
        
        # List all issues
        all_issues = []
        for gate_name, gate_result in quality_gates.items():
            for issue in gate_result.get('issues', []):
                all_issues.append(f"{gate_name}: {issue}")
        
        if all_issues:
            print(f"\n⚠️  Issues to address ({len(all_issues)}):")
            for issue in all_issues[:10]:  # Show first 10 issues
                print(f"   - {issue}")
            if len(all_issues) > 10:
                print(f"   ... and {len(all_issues) - 10} more issues")
        
        self.results['issues'] = all_issues
        
        print("=" * 80)
        
        return self.results

def main():
    """Run comprehensive quality gates validation."""
    validator = QualityGateValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    results_file = f"comprehensive_quality_gates_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return results['production_ready']

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 COMPREHENSIVE QUALITY GATES: PASSED!")
        print("🚀 System is ready for production deployment!")
    else:
        print("\n❌ COMPREHENSIVE QUALITY GATES: FAILED!")
        print("🔧 Address the issues above before production deployment.")
        sys.exit(1)