"""Autonomous SDLC Quality Gates Implementation.

This module implements comprehensive quality gates that automatically validate
the entire SDLC implementation including code quality, performance benchmarks,
security checks, and production readiness assessments.
"""

import os
import sys
import subprocess
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from contextlib import contextmanager


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    execution_time_seconds: float
    recommendations: List[str]


class QualityGateRunner:
    """Orchestrates all quality gate checks."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.logger = logging.getLogger("QualityGateRunner")
        self.results = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        
        self.logger.info("Starting Autonomous SDLC Quality Gate Execution")
        
        quality_gates = [
            self.gate_1_code_structure_validation,
            self.gate_2_import_and_syntax_validation,
            self.gate_3_functional_testing,
            self.gate_4_performance_benchmarking,
            self.gate_5_security_scanning,
            self.gate_6_documentation_quality,
            self.gate_7_research_validation,
            self.gate_8_integration_testing,
            self.gate_9_production_readiness,
            self.gate_10_final_validation
        ]
        
        self.results = []
        overall_start_time = time.time()
        
        for gate_func in quality_gates:
            self.logger.info(f"Executing {gate_func.__name__}")
            
            try:
                result = gate_func()
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                self.logger.info(f"{gate_func.__name__}: {status} (Score: {result.score:.2f})")
                
            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate_func.__name__,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    errors=[f"Gate execution failed: {str(e)}"],
                    warnings=[],
                    execution_time_seconds=0.0,
                    recommendations=[f"Fix gate execution error: {str(e)}"]
                )
                self.results.append(error_result)
                self.logger.error(f"{gate_func.__name__}: EXECUTION ERROR - {str(e)}")
        
        total_execution_time = time.time() - overall_start_time
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(total_execution_time)
        
        return report
    
    def gate_1_code_structure_validation(self) -> QualityGateResult:
        """Validate overall code structure and organization."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Check directory structure
        expected_dirs = [
            'src/pno_physics_bench',
            'src/pno_physics_bench/research',
            'src/pno_physics_bench/robustness',
            'src/pno_physics_bench/validation',
            'src/pno_physics_bench/monitoring',
            'src/pno_physics_bench/scaling',
            'src/pno_physics_bench/optimization',
            'tests',
            'docs',
            'deployment'
        ]
        
        missing_dirs = []
        present_dirs = []
        
        for dir_path in expected_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                present_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        details['present_directories'] = present_dirs
        details['missing_directories'] = missing_dirs
        
        # Check for required files
        required_files = [
            'README.md',
            'pyproject.toml',
            'requirements.txt',
            'src/pno_physics_bench/__init__.py'
        ]
        
        missing_files = []
        present_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                present_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        details['present_files'] = present_files
        details['missing_files'] = missing_files
        
        # Count Python files
        src_path = self.project_root / 'src'
        if src_path.exists():
            python_files = list(src_path.rglob('*.py'))
            details['python_file_count'] = len(python_files)
            details['lines_of_code'] = self._count_lines_of_code(python_files)
        else:
            details['python_file_count'] = 0
            details['lines_of_code'] = 0
        
        # Calculate score
        structure_score = len(present_dirs) / len(expected_dirs)
        files_score = len(present_files) / len(required_files)
        overall_score = (structure_score + files_score) / 2
        
        # Generate recommendations
        if missing_dirs:
            recommendations.extend([f"Create missing directory: {d}" for d in missing_dirs])
        if missing_files:
            recommendations.extend([f"Create missing file: {f}" for f in missing_files])
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Code Structure Validation",
            passed=overall_score >= 0.8,
            score=overall_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_2_import_and_syntax_validation(self) -> QualityGateResult:
        """Validate that all modules can be imported and have valid syntax."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Find all Python files
        src_path = self.project_root / 'src'
        if not src_path.exists():
            return QualityGateResult(
                gate_name="Import and Syntax Validation",
                passed=False,
                score=0.0,
                details={'error': 'src directory not found'},
                errors=['src directory not found'],
                warnings=[],
                execution_time_seconds=time.time() - start_time,
                recommendations=['Create src directory structure']
            )
        
        python_files = list(src_path.rglob('*.py'))
        
        # Test syntax validation
        syntax_valid = []
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                compile(source, py_file, 'exec')
                syntax_valid.append(str(py_file.relative_to(src_path)))
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file.relative_to(src_path)}: {str(e)}")
            except Exception as e:
                warnings.append(f"Could not check syntax for {py_file.relative_to(src_path)}: {str(e)}")
        
        details['syntax_valid_files'] = syntax_valid
        details['syntax_errors'] = syntax_errors
        details['total_files_checked'] = len(python_files)
        
        # Test imports (basic check)
        import_successful = []
        import_failures = []
        
        # Add src to Python path
        sys.path.insert(0, str(src_path))
        
        # Test key module imports
        key_modules = [
            'pno_physics_bench',
            'pno_physics_bench.research.adaptive_uncertainty_calibration',
            'pno_physics_bench.research.quantum_enhanced_uncertainty',
            'pno_physics_bench.research.continual_learning_uncertainty',
            'pno_physics_bench.robustness.advanced_error_handling',
            'pno_physics_bench.validation.comprehensive_input_validation',
            'pno_physics_bench.monitoring.comprehensive_system_monitoring',
            'pno_physics_bench.scaling.distributed_inference_optimization',
            'pno_physics_bench.scaling.memory_efficient_training',
            'pno_physics_bench.optimization.advanced_performance_optimization'
        ]
        
        for module_name in key_modules:
            try:
                __import__(module_name)
                import_successful.append(module_name)
            except ImportError as e:
                import_failures.append(f"{module_name}: {str(e)}")
            except Exception as e:
                warnings.append(f"Unexpected error importing {module_name}: {str(e)}")
        
        details['import_successful'] = import_successful
        details['import_failures'] = import_failures
        
        # Calculate score
        syntax_score = len(syntax_valid) / len(python_files) if python_files else 0
        import_score = len(import_successful) / len(key_modules) if key_modules else 0
        overall_score = (syntax_score + import_score) / 2
        
        # Add errors and recommendations
        if syntax_errors:
            errors.extend(syntax_errors)
            recommendations.append("Fix syntax errors in Python files")
        
        if import_failures:
            errors.extend(import_failures)
            recommendations.append("Fix import errors and missing dependencies")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Import and Syntax Validation",
            passed=overall_score >= 0.9 and len(syntax_errors) == 0,
            score=overall_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_3_functional_testing(self) -> QualityGateResult:
        """Run comprehensive functional tests."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Run the comprehensive test suite
        test_script_path = self.project_root / 'test_generation_enhancements_comprehensive.py'
        
        if not test_script_path.exists():
            errors.append("Comprehensive test suite not found")
            return QualityGateResult(
                gate_name="Functional Testing",
                passed=False,
                score=0.0,
                details={'error': 'Test suite not found'},
                errors=errors,
                warnings=warnings,
                execution_time_seconds=time.time() - start_time,
                recommendations=['Create comprehensive test suite']
            )
        
        try:
            # Execute the test script
            result = subprocess.run([
                sys.executable, str(test_script_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            details['test_exit_code'] = result.returncode
            details['test_stdout'] = result.stdout
            details['test_stderr'] = result.stderr
            
            # Parse test results from output
            test_passed = result.returncode == 0
            
            # Extract metrics from test output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'tests passed' in line.lower():
                    details['test_summary'] = line.strip()
                elif 'success rate' in line.lower():
                    details['success_rate'] = line.strip()
            
            if not test_passed:
                errors.append("Functional tests failed")
                recommendations.append("Fix failing functional tests")
            
            # Calculate score based on test results
            if test_passed:
                score = 1.0
            else:
                # Try to extract partial success rate
                try:
                    # Look for success rate in output
                    for line in output_lines:
                        if 'success rate:' in line.lower():
                            rate_str = line.split(':')[-1].strip().replace('%', '')
                            score = float(rate_str) / 100.0
                            break
                    else:
                        score = 0.0
                except:
                    score = 0.0
            
        except subprocess.TimeoutExpired:
            errors.append("Test execution timed out")
            recommendations.append("Optimize test performance or increase timeout")
            score = 0.0
            details['timeout'] = True
            
        except Exception as e:
            errors.append(f"Test execution failed: {str(e)}")
            recommendations.append("Fix test execution environment")
            score = 0.0
            details['execution_error'] = str(e)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Functional Testing",
            passed=score >= 0.85,  # Require 85% test success
            score=score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_4_performance_benchmarking(self) -> QualityGateResult:
        """Run performance benchmarks and validate performance requirements."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        try:
            # Add src to path
            sys.path.insert(0, str(self.project_root / 'src'))
            
            # Import performance optimization module
            from pno_physics_bench.optimization.advanced_performance_optimization import (
                create_performance_optimized_model,
                PerformanceProfiler
            )
            
            # Create test model
            model = create_performance_optimized_model(
                input_dim=64,
                hidden_dim=128,
                num_layers=2,
                enable_auto_optimization=True
            )
            
            # Run performance benchmarks
            profiler = PerformanceProfiler(enable_detailed_profiling=True)
            
            # Benchmark inference performance
            batch_sizes = [1, 8, 32, 64]
            benchmark_results = {}
            
            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, 64)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(test_input)
                
                # Actual benchmark
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_bench = time.time()
                
                with profiler.profile_operation(f"inference_batch_{batch_size}", batch_size):
                    with torch.no_grad():
                        for _ in range(10):
                            output = model(test_input)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_bench = time.time()
                
                avg_time_ms = ((end_bench - start_bench) / 10) * 1000
                throughput = batch_size / (avg_time_ms / 1000)
                
                benchmark_results[f"batch_{batch_size}"] = {
                    'avg_inference_time_ms': avg_time_ms,
                    'throughput_samples_per_sec': throughput
                }
            
            details['benchmark_results'] = benchmark_results
            
            # Get detailed performance report
            perf_report = profiler.get_performance_report()
            details['performance_report'] = perf_report
            
            # Performance requirements validation
            requirements = {
                'single_inference_time_ms': 100.0,  # Max 100ms for single sample
                'batch_32_throughput': 50.0,  # Min 50 samples/sec for batch of 32
                'memory_efficiency': 500.0  # Max 500MB for model
            }
            
            violations = []
            
            # Check single inference time
            single_time = benchmark_results['batch_1']['avg_inference_time_ms']
            if single_time > requirements['single_inference_time_ms']:
                violations.append(f"Single inference time {single_time:.1f}ms exceeds {requirements['single_inference_time_ms']}ms")
            
            # Check batch throughput
            if 'batch_32' in benchmark_results:
                batch_throughput = benchmark_results['batch_32']['throughput_samples_per_sec']
                if batch_throughput < requirements['batch_32_throughput']:
                    violations.append(f"Batch throughput {batch_throughput:.1f} below {requirements['batch_32_throughput']}")
            
            details['requirement_violations'] = violations
            
            # Calculate performance score
            score_components = []
            
            # Inference time score (inverse relationship)
            time_score = min(1.0, requirements['single_inference_time_ms'] / single_time)
            score_components.append(time_score)
            
            # Throughput score
            if 'batch_32' in benchmark_results:
                throughput_score = min(1.0, batch_throughput / requirements['batch_32_throughput'])
                score_components.append(throughput_score)
            
            overall_score = np.mean(score_components)
            
            if violations:
                errors.extend(violations)
                recommendations.append("Optimize model performance to meet requirements")
            
        except ImportError as e:
            errors.append(f"Performance modules not available: {str(e)}")
            recommendations.append("Ensure performance optimization modules are properly implemented")
            overall_score = 0.0
            
        except Exception as e:
            errors.append(f"Performance benchmarking failed: {str(e)}")
            recommendations.append("Fix performance benchmarking implementation")
            overall_score = 0.0
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Performance Benchmarking",
            passed=overall_score >= 0.7 and len(violations) == 0,
            score=overall_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_5_security_scanning(self) -> QualityGateResult:
        """Perform security scanning and validation."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Check for security-sensitive patterns in code
        security_checks = {
            'hardcoded_secrets': 0,
            'eval_usage': 0,
            'exec_usage': 0,
            'pickle_usage': 0,
            'shell_injection_risk': 0,
            'path_traversal_risk': 0
        }
        
        src_path = self.project_root / 'src'
        if src_path.exists():
            python_files = list(src_path.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for security anti-patterns
                    if 'eval(' in content:
                        security_checks['eval_usage'] += content.count('eval(')
                    
                    if 'exec(' in content:
                        security_checks['exec_usage'] += content.count('exec(')
                    
                    if 'pickle.' in content or 'cPickle' in content:
                        security_checks['pickle_usage'] += 1
                    
                    # Check for potential hardcoded secrets
                    secret_patterns = ['password', 'secret', 'token', 'api_key', 'private_key']
                    for pattern in secret_patterns:
                        if f'{pattern} =' in content.lower():
                            security_checks['hardcoded_secrets'] += 1
                    
                    # Check for shell command risks
                    if 'subprocess.call' in content or 'os.system' in content:
                        security_checks['shell_injection_risk'] += 1
                    
                    # Check for path traversal risks
                    if '../' in content or '..\\' in content:
                        security_checks['path_traversal_risk'] += 1
                
                except Exception as e:
                    warnings.append(f"Could not scan {py_file}: {str(e)}")
        
        details['security_scan_results'] = security_checks
        
        # Evaluate security posture
        security_violations = []
        
        for check, count in security_checks.items():
            if count > 0:
                if check in ['eval_usage', 'exec_usage']:
                    security_violations.append(f"Found {count} instances of {check}")
                elif check == 'hardcoded_secrets':
                    security_violations.append(f"Potential hardcoded secrets: {count}")
                else:
                    warnings.append(f"Security concern - {check}: {count}")
        
        details['security_violations'] = security_violations
        
        # Check for security-related dependencies
        requirements_file = self.project_root / 'requirements.txt'
        security_packages = []
        
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                requirements = f.read()
                
            # Look for security-related packages
            security_indicators = ['cryptography', 'bcrypt', 'security', 'ssl']
            for indicator in security_indicators:
                if indicator in requirements.lower():
                    security_packages.append(indicator)
        
        details['security_packages'] = security_packages
        
        # Calculate security score
        base_score = 1.0
        
        # Deduct for violations
        for violation in security_violations:
            if 'eval_usage' in violation or 'exec_usage' in violation:
                base_score -= 0.3  # Major deduction
            else:
                base_score -= 0.1
        
        # Bonus for security packages
        if security_packages:
            base_score += 0.1
        
        security_score = max(0.0, min(1.0, base_score))
        
        if security_violations:
            errors.extend(security_violations)
            recommendations.append("Address security violations in code")
        
        if not security_packages:
            recommendations.append("Consider adding security-focused dependencies")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Security Scanning",
            passed=security_score >= 0.8 and len(security_violations) == 0,
            score=security_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_6_documentation_quality(self) -> QualityGateResult:
        """Assess documentation quality and completeness."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Check README quality
        readme_path = self.project_root / 'README.md'
        readme_score = 0.0
        
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            readme_checks = {
                'has_title': bool(readme_content.startswith('#')),
                'has_description': len(readme_content) > 100,
                'has_installation': 'install' in readme_content.lower(),
                'has_usage_examples': 'example' in readme_content.lower() or '```' in readme_content,
                'has_features': 'feature' in readme_content.lower(),
                'has_contributing': 'contribut' in readme_content.lower()
            }
            
            readme_score = sum(readme_checks.values()) / len(readme_checks)
            details['readme_quality'] = readme_checks
            
        else:
            errors.append("README.md file not found")
            recommendations.append("Create comprehensive README.md")
        
        # Check docstring coverage
        src_path = self.project_root / 'src'
        docstring_stats = {
            'modules_with_docstrings': 0,
            'total_modules': 0,
            'functions_with_docstrings': 0,
            'total_functions': 0,
            'classes_with_docstrings': 0,
            'total_classes': 0
        }
        
        if src_path.exists():
            python_files = list(src_path.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    docstring_stats['total_modules'] += 1
                    
                    # Check for module docstring
                    if '"""' in content[:500] or "'''" in content[:500]:
                        docstring_stats['modules_with_docstrings'] += 1
                    
                    # Count functions and their docstrings
                    import ast
                    try:
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                docstring_stats['total_functions'] += 1
                                if ast.get_docstring(node):
                                    docstring_stats['functions_with_docstrings'] += 1
                            
                            elif isinstance(node, ast.ClassDef):
                                docstring_stats['total_classes'] += 1
                                if ast.get_docstring(node):
                                    docstring_stats['classes_with_docstrings'] += 1
                    
                    except SyntaxError:
                        pass  # Skip files with syntax errors
                
                except Exception as e:
                    warnings.append(f"Could not analyze docstrings in {py_file}: {str(e)}")
        
        details['docstring_statistics'] = docstring_stats
        
        # Calculate documentation scores
        module_doc_score = (docstring_stats['modules_with_docstrings'] / 
                           max(1, docstring_stats['total_modules']))
        
        function_doc_score = (docstring_stats['functions_with_docstrings'] / 
                             max(1, docstring_stats['total_functions']))
        
        class_doc_score = (docstring_stats['classes_with_docstrings'] / 
                          max(1, docstring_stats['total_classes']))
        
        # Check for additional documentation
        docs_path = self.project_root / 'docs'
        has_docs_dir = docs_path.exists()
        
        additional_docs = []
        if has_docs_dir:
            doc_files = list(docs_path.rglob('*.md')) + list(docs_path.rglob('*.rst'))
            additional_docs = [str(f.relative_to(docs_path)) for f in doc_files]
        
        details['additional_documentation'] = additional_docs
        
        # Overall documentation score
        scores = [readme_score, module_doc_score, function_doc_score, class_doc_score]
        if has_docs_dir:
            scores.append(0.8)  # Bonus for having docs directory
        
        overall_doc_score = np.mean(scores)
        
        # Generate recommendations
        if readme_score < 0.7:
            recommendations.append("Improve README.md quality and completeness")
        
        if function_doc_score < 0.5:
            recommendations.append("Add docstrings to functions and methods")
        
        if class_doc_score < 0.5:
            recommendations.append("Add docstrings to classes")
        
        if not has_docs_dir:
            recommendations.append("Create comprehensive documentation directory")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Documentation Quality",
            passed=overall_doc_score >= 0.6,
            score=overall_doc_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_7_research_validation(self) -> QualityGateResult:
        """Validate research implementations and novel algorithms."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        research_modules = [
            'adaptive_uncertainty_calibration',
            'quantum_enhanced_uncertainty', 
            'continual_learning_uncertainty'
        ]
        
        research_validation_results = {}
        
        try:
            sys.path.insert(0, str(self.project_root / 'src'))
            
            for module_name in research_modules:
                module_results = {
                    'importable': False,
                    'key_classes_present': False,
                    'functional_test_passed': False,
                    'algorithmic_correctness': False
                }
                
                try:
                    # Test import
                    full_module_name = f'pno_physics_bench.research.{module_name}'
                    module = __import__(full_module_name, fromlist=[''])
                    module_results['importable'] = True
                    
                    # Check for key classes/functions
                    if module_name == 'adaptive_uncertainty_calibration':
                        required_items = ['AdaptiveUncertaintyCalibrator', 'HierarchicalUncertaintyDecomposer']
                    elif module_name == 'quantum_enhanced_uncertainty':
                        required_items = ['QuantumStatePreparation', 'QuantumUncertaintyNeuralOperator']
                    elif module_name == 'continual_learning_uncertainty':
                        required_items = ['ElasticWeightConsolidation', 'UncertaintyAwareContinualLearner']
                    else:
                        required_items = []
                    
                    present_items = []
                    for item in required_items:
                        if hasattr(module, item):
                            present_items.append(item)
                    
                    module_results['key_classes_present'] = len(present_items) == len(required_items)
                    module_results['present_items'] = present_items
                    
                    # Basic functional test
                    if module_name == 'adaptive_uncertainty_calibration' and module_results['key_classes_present']:
                        calibrator = module.AdaptiveUncertaintyCalibrator(input_dim=32)
                        test_pred = torch.randn(4, 16)
                        test_unc = torch.rand(4, 16)
                        result = calibrator(test_pred, test_unc)
                        module_results['functional_test_passed'] = len(result) == 2
                    
                    elif module_name == 'quantum_enhanced_uncertainty' and module_results['key_classes_present']:
                        quantum_prep = module.QuantumStatePreparation(num_qubits=3, feature_dim=16)
                        features = torch.randn(2, 16)
                        quantum_state = quantum_prep.prepare_quantum_state(features)
                        module_results['functional_test_passed'] = quantum_state.shape[1] == 8  # 2^3
                    
                    elif module_name == 'continual_learning_uncertainty' and module_results['key_classes_present']:
                        from pno_physics_bench.research.continual_learning_uncertainty import EpisodicMemory
                        memory = module.EpisodicMemory(max_size=10)
                        sample = torch.randn(16)
                        target = torch.randn(8)
                        memory.add_sample(sample, target, 0)
                        module_results['functional_test_passed'] = len(memory) == 1
                    
                    # Algorithmic correctness (basic sanity checks)
                    module_results['algorithmic_correctness'] = (
                        module_results['importable'] and 
                        module_results['key_classes_present'] and
                        module_results['functional_test_passed']
                    )
                    
                except Exception as e:
                    module_results['error'] = str(e)
                    warnings.append(f"Research module {module_name} validation failed: {str(e)}")
                
                research_validation_results[module_name] = module_results
            
            details['research_validation_results'] = research_validation_results
            
            # Calculate research score
            total_modules = len(research_modules)
            successful_modules = sum(1 for results in research_validation_results.values() 
                                   if results.get('algorithmic_correctness', False))
            
            research_score = successful_modules / total_modules
            
            if research_score < 1.0:
                failed_modules = [name for name, results in research_validation_results.items() 
                                if not results.get('algorithmic_correctness', False)]
                errors.append(f"Research modules failed validation: {failed_modules}")
                recommendations.append("Fix failing research module implementations")
        
        except Exception as e:
            errors.append(f"Research validation setup failed: {str(e)}")
            research_score = 0.0
            recommendations.append("Fix research validation environment")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Research Validation",
            passed=research_score >= 0.8,
            score=research_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_8_integration_testing(self) -> QualityGateResult:
        """Test integration between different components."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        integration_tests = {
            'robustness_with_validation': False,
            'monitoring_with_training': False,
            'scaling_with_optimization': False,
            'research_with_core_models': False
        }
        
        try:
            sys.path.insert(0, str(self.project_root / 'src'))
            
            # Test 1: Robustness with Validation
            try:
                from pno_physics_bench.robustness.advanced_error_handling import RobustPNOWrapper
                from pno_physics_bench.validation.comprehensive_input_validation import validate_pno_input
                
                # Create a simple mock model
                class SimpleModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = torch.nn.Linear(32, 32)
                    def forward(self, x):
                        return self.linear(x)
                    def predict_with_uncertainty(self, x):
                        return self.forward(x), torch.ones_like(x) * 0.1
                
                model = SimpleModel()
                robust_model = RobustPNOWrapper(model)
                
                # Test with validation
                test_input = torch.randn(4, 32)
                validation_result = validate_pno_input(test_input)
                
                if validation_result.is_valid:
                    output = robust_model(validation_result.sanitized_input)
                    integration_tests['robustness_with_validation'] = output.shape == (4, 32)
                
            except Exception as e:
                warnings.append(f"Robustness-Validation integration failed: {str(e)}")
            
            # Test 2: Monitoring with Training  
            try:
                from pno_physics_bench.monitoring.comprehensive_system_monitoring import ComprehensiveMonitor
                
                model = SimpleModel()
                monitor = ComprehensiveMonitor(model=model, enable_real_time=False)
                
                # Simulate training step
                inputs = torch.randn(8, 32)
                predictions, uncertainties = model.predict_with_uncertainty(inputs)
                targets = torch.randn(8, 32)
                
                monitor.record_inference(inputs, predictions, uncertainties, targets, 0.1)
                health_report = monitor.get_system_health_report()
                
                integration_tests['monitoring_with_training'] = isinstance(health_report, dict)
                
            except Exception as e:
                warnings.append(f"Monitoring-Training integration failed: {str(e)}")
            
            # Test 3: Research with Core Models
            try:
                from pno_physics_bench.research.adaptive_uncertainty_calibration import AdaptiveUncertaintyCalibrator
                
                calibrator = AdaptiveUncertaintyCalibrator(input_dim=64, hidden_dim=32)
                
                # Test with model outputs
                model_predictions = torch.randn(8, 32)
                model_uncertainties = torch.rand(8, 32) * 0.3
                
                calibrated_uncertainties, params = calibrator(model_predictions, model_uncertainties)
                
                integration_tests['research_with_core_models'] = (
                    calibrated_uncertainties.shape == model_uncertainties.shape and
                    torch.all(calibrated_uncertainties >= 0)
                )
                
            except Exception as e:
                warnings.append(f"Research-Core integration failed: {str(e)}")
            
        except Exception as e:
            errors.append(f"Integration testing setup failed: {str(e)}")
        
        details['integration_test_results'] = integration_tests
        
        # Calculate integration score
        passed_tests = sum(integration_tests.values())
        total_tests = len(integration_tests)
        integration_score = passed_tests / total_tests
        
        if integration_score < 1.0:
            failed_tests = [test for test, passed in integration_tests.items() if not passed]
            errors.append(f"Integration tests failed: {failed_tests}")
            recommendations.append("Fix component integration issues")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Integration Testing",
            passed=integration_score >= 0.75,
            score=integration_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_9_production_readiness(self) -> QualityGateResult:
        """Assess production readiness and deployment preparedness."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        production_criteria = {
            'deployment_files_present': False,
            'configuration_management': False,
            'logging_implementation': False,
            'monitoring_setup': False,
            'error_handling_comprehensive': False,
            'scalability_features': False,
            'security_measures': False
        }
        
        # Check deployment files
        deployment_path = self.project_root / 'deployment'
        if deployment_path.exists():
            deployment_files = list(deployment_path.rglob('*'))
            production_criteria['deployment_files_present'] = len(deployment_files) > 0
            details['deployment_files'] = [str(f.relative_to(deployment_path)) for f in deployment_files]
        
        # Check for Docker files
        docker_files = ['Dockerfile', 'docker-compose.yml']
        present_docker_files = []
        for docker_file in docker_files:
            if (self.project_root / docker_file).exists():
                present_docker_files.append(docker_file)
        
        details['docker_files'] = present_docker_files
        production_criteria['deployment_files_present'] = (
            production_criteria['deployment_files_present'] or len(present_docker_files) > 0
        )
        
        # Check configuration management
        config_files = ['pyproject.toml', 'requirements.txt']
        present_config_files = []
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                present_config_files.append(config_file)
        
        details['configuration_files'] = present_config_files
        production_criteria['configuration_management'] = len(present_config_files) >= 2
        
        # Check for logging implementation
        src_path = self.project_root / 'src'
        if src_path.exists():
            python_files = list(src_path.rglob('*.py'))
            logging_usage = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'import logging' in content or 'from logging' in content:
                        logging_usage += 1
                
                except Exception:
                    pass
            
            production_criteria['logging_implementation'] = logging_usage > 0
            details['files_with_logging'] = logging_usage
        
        # Check monitoring implementation
        monitoring_path = self.project_root / 'src' / 'pno_physics_bench' / 'monitoring'
        if monitoring_path.exists():
            monitoring_files = list(monitoring_path.glob('*.py'))
            production_criteria['monitoring_setup'] = len(monitoring_files) > 0
            details['monitoring_modules'] = len(monitoring_files)
        
        # Check error handling
        robustness_path = self.project_root / 'src' / 'pno_physics_bench' / 'robustness'
        if robustness_path.exists():
            robustness_files = list(robustness_path.glob('*.py'))
            production_criteria['error_handling_comprehensive'] = len(robustness_files) > 0
            details['error_handling_modules'] = len(robustness_files)
        
        # Check scalability features
        scaling_path = self.project_root / 'src' / 'pno_physics_bench' / 'scaling'
        if scaling_path.exists():
            scaling_files = list(scaling_path.glob('*.py'))
            production_criteria['scalability_features'] = len(scaling_files) > 0
            details['scaling_modules'] = len(scaling_files)
        
        # Check security implementation
        validation_path = self.project_root / 'src' / 'pno_physics_bench' / 'validation'
        if validation_path.exists():
            validation_files = list(validation_path.glob('*.py'))
            production_criteria['security_measures'] = len(validation_files) > 0
            details['security_modules'] = len(validation_files)
        
        details['production_criteria'] = production_criteria
        
        # Calculate production readiness score
        passed_criteria = sum(production_criteria.values())
        total_criteria = len(production_criteria)
        production_score = passed_criteria / total_criteria
        
        # Generate recommendations
        for criterion, passed in production_criteria.items():
            if not passed:
                if criterion == 'deployment_files_present':
                    recommendations.append("Create deployment files (Dockerfile, deployment manifests)")
                elif criterion == 'configuration_management':
                    recommendations.append("Improve configuration management (requirements, settings)")
                elif criterion == 'logging_implementation':
                    recommendations.append("Add comprehensive logging throughout codebase")
                elif criterion == 'monitoring_setup':
                    recommendations.append("Implement monitoring and observability")
                elif criterion == 'error_handling_comprehensive':
                    recommendations.append("Add comprehensive error handling and recovery")
                elif criterion == 'scalability_features':
                    recommendations.append("Implement scalability and performance optimizations")
                elif criterion == 'security_measures':
                    recommendations.append("Add security measures and input validation")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Production Readiness",
            passed=production_score >= 0.8,
            score=production_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def gate_10_final_validation(self) -> QualityGateResult:
        """Final comprehensive validation of the entire system."""
        
        start_time = time.time()
        details = {}
        errors = []
        warnings = []
        recommendations = []
        
        # Aggregate results from all previous gates
        previous_gates = [result for result in self.results if result.gate_name != "Final Validation"]
        
        gate_scores = {}
        gate_status = {}
        
        for gate_result in previous_gates:
            gate_scores[gate_result.gate_name] = gate_result.score
            gate_status[gate_result.gate_name] = gate_result.passed
        
        details['individual_gate_scores'] = gate_scores
        details['individual_gate_status'] = gate_status
        
        # Calculate weighted overall score
        gate_weights = {
            "Code Structure Validation": 0.10,
            "Import and Syntax Validation": 0.15,
            "Functional Testing": 0.20,
            "Performance Benchmarking": 0.15,
            "Security Scanning": 0.10,
            "Documentation Quality": 0.08,
            "Research Validation": 0.12,
            "Integration Testing": 0.08,
            "Production Readiness": 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for gate_name, score in gate_scores.items():
            weight = gate_weights.get(gate_name, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        details['weighted_final_score'] = final_score
        
        # Critical gate requirements
        critical_gates = [
            "Import and Syntax Validation",
            "Functional Testing",
            "Security Scanning"
        ]
        
        critical_failures = []
        for gate_name in critical_gates:
            if not gate_status.get(gate_name, False):
                critical_failures.append(gate_name)
        
        details['critical_failures'] = critical_failures
        
        # Overall system assessment
        system_assessment = {
            'code_quality': gate_scores.get("Code Structure Validation", 0) * 0.5 + 
                           gate_scores.get("Import and Syntax Validation", 0) * 0.5,
            'functionality': gate_scores.get("Functional Testing", 0),
            'performance': gate_scores.get("Performance Benchmarking", 0),
            'security': gate_scores.get("Security Scanning", 0),
            'research_innovation': gate_scores.get("Research Validation", 0),
            'production_readiness': gate_scores.get("Production Readiness", 0)
        }
        
        details['system_assessment'] = system_assessment
        
        # Pass/fail determination
        final_passed = (
            final_score >= 0.75 and  # Overall score requirement
            len(critical_failures) == 0 and  # No critical gate failures
            all(score >= 0.5 for score in system_assessment.values())  # All areas meet minimum
        )
        
        # Generate final recommendations
        if critical_failures:
            errors.append(f"Critical quality gates failed: {critical_failures}")
            recommendations.append("Address all critical quality gate failures before deployment")
        
        if final_score < 0.75:
            recommendations.append("Improve overall system quality to achieve 75% minimum score")
        
        for area, score in system_assessment.items():
            if score < 0.6:
                recommendations.append(f"Improve {area} (current score: {score:.2f})")
        
        if final_passed:
            recommendations.append("System ready for production deployment! 🎉")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Final Validation",
            passed=final_passed,
            score=final_score,
            details=details,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
            recommendations=recommendations
        )
    
    def _count_lines_of_code(self, python_files: List[Path]) -> int:
        """Count total lines of code."""
        total_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Count non-empty, non-comment lines
                    code_lines = [line for line in lines 
                                if line.strip() and not line.strip().startswith('#')]
                    total_lines += len(code_lines)
            except Exception:
                pass
        
        return total_lines
    
    def generate_comprehensive_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality gates report."""
        
        # Summary statistics
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results if result.passed)
        average_score = np.mean([result.score for result in self.results])
        
        # Categorize results
        excellent_gates = [r for r in self.results if r.score >= 0.9]
        good_gates = [r for r in self.results if 0.7 <= r.score < 0.9]
        needs_improvement = [r for r in self.results if 0.5 <= r.score < 0.7]
        failing_gates = [r for r in self.results if r.score < 0.5]
        
        # Collect all errors and recommendations
        all_errors = []
        all_recommendations = []
        
        for result in self.results:
            all_errors.extend(result.errors)
            all_recommendations.extend(result.recommendations)
        
        # Create comprehensive report
        report = {
            'execution_metadata': {
                'timestamp': time.time(),
                'total_execution_time_seconds': total_execution_time,
                'project_root': str(self.project_root),
                'total_quality_gates': total_gates
            },
            
            'summary': {
                'overall_success': passed_gates == total_gates and average_score >= 0.75,
                'gates_passed': passed_gates,
                'gates_failed': total_gates - passed_gates,
                'success_rate': passed_gates / total_gates if total_gates > 0 else 0,
                'average_score': average_score,
                'total_errors': len(all_errors),
                'total_recommendations': len(all_recommendations)
            },
            
            'gate_categories': {
                'excellent': [{'name': r.gate_name, 'score': r.score} for r in excellent_gates],
                'good': [{'name': r.gate_name, 'score': r.score} for r in good_gates],
                'needs_improvement': [{'name': r.gate_name, 'score': r.score} for r in needs_improvement],
                'failing': [{'name': r.gate_name, 'score': r.score} for r in failing_gates]
            },
            
            'detailed_results': [asdict(result) for result in self.results],
            
            'consolidated_errors': list(set(all_errors)),
            'consolidated_recommendations': list(set(all_recommendations)),
            
            'next_steps': self._generate_next_steps(passed_gates == total_gates, average_score)
        }
        
        return report
    
    def _generate_next_steps(self, all_passed: bool, average_score: float) -> List[str]:
        """Generate next steps based on quality gate results."""
        
        next_steps = []
        
        if all_passed and average_score >= 0.9:
            next_steps = [
                "🎉 Excellent! All quality gates passed with high scores",
                "✅ System is ready for production deployment",
                "🚀 Consider advanced optimizations and performance tuning",
                "📊 Set up continuous monitoring and alerting",
                "🔄 Establish regular quality gate runs in CI/CD pipeline"
            ]
        
        elif all_passed and average_score >= 0.75:
            next_steps = [
                "✅ Good! All quality gates passed",
                "⚡ Focus on improving lower-scoring areas",
                "🚀 System is ready for production with monitoring",
                "🔧 Address specific recommendations from individual gates",
                "📈 Implement performance monitoring in production"
            ]
        
        elif average_score >= 0.6:
            next_steps = [
                "⚠️ Some quality gates need attention",
                "🔧 Address failing gates before production deployment",
                "📊 Focus on critical areas: security, functionality, performance",
                "🧪 Run additional testing for failing components",
                "🔄 Re-run quality gates after fixes"
            ]
        
        else:
            next_steps = [
                "❌ Significant quality issues detected",
                "🛑 Do not deploy to production until issues are resolved",
                "🔧 Focus on critical failures first",
                "🧪 Implement comprehensive testing",
                "📚 Review implementation against best practices",
                "🔄 Re-run all quality gates after major fixes"
            ]
        
        return next_steps


def main():
    """Main execution function for quality gates."""
    
    print("🚀 AUTONOMOUS SDLC - QUALITY GATES EXECUTION")
    print("=" * 80)
    
    # Initialize quality gate runner
    runner = QualityGateRunner()
    
    # Run all quality gates
    report = runner.run_all_quality_gates()
    
    # Print summary
    print("\n📊 QUALITY GATES EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Gates Passed: {report['summary']['gates_passed']}/{report['summary']['total_quality_gates']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Average Score: {report['summary']['average_score']:.3f}")
    print(f"Overall Success: {'✅ YES' if report['summary']['overall_success'] else '❌ NO'}")
    
    # Print category breakdown
    print(f"\n📈 PERFORMANCE BREAKDOWN")
    print(f"Excellent (≥90%): {len(report['gate_categories']['excellent'])} gates")
    print(f"Good (70-89%): {len(report['gate_categories']['good'])} gates")
    print(f"Needs Improvement (50-69%): {len(report['gate_categories']['needs_improvement'])} gates")
    print(f"Failing (<50%): {len(report['gate_categories']['failing'])} gates")
    
    # Print next steps
    print(f"\n🎯 NEXT STEPS")
    for i, step in enumerate(report['next_steps'], 1):
        print(f"{i}. {step}")
    
    # Save detailed report
    report_path = Path("autonomous_sdlc_quality_gates_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Exit code
    exit_code = 0 if report['summary']['overall_success'] else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)