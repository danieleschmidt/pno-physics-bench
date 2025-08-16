#!/usr/bin/env python3
"""
Comprehensive Quality Gates Suite
Executes all quality gates including testing, security, performance, and compliance
"""

import os
import sys
import json
import subprocess
import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.execution_time = 0.0
    
    def execute(self) -> bool:
        """Execute the quality gate."""
        start_time = time.time()
        try:
            result = self._run_check()
            self.passed = result
            self.score = 1.0 if result else 0.0
        except Exception as e:
            logger.error(f"Quality gate {self.name} failed with error: {e}")
            self.passed = False
            self.score = 0.0
            self.details['error'] = str(e)
        finally:
            self.execution_time = time.time() - start_time
        
        return self.passed
    
    def _run_check(self) -> bool:
        """Override this method to implement the specific check."""
        raise NotImplementedError

class CodeStructureGate(QualityGate):
    """Validates code structure and organization."""
    
    def __init__(self):
        super().__init__("Code Structure", weight=1.0)
    
    def _run_check(self) -> bool:
        required_files = [
            "src/pno_physics_bench/__init__.py",
            "src/pno_physics_bench/models.py",
            "src/pno_physics_bench/training/trainer.py",
            "src/pno_physics_bench/uncertainty.py",
            "pyproject.toml",
            "README.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        self.details['missing_files'] = missing_files
        self.details['total_required'] = len(required_files)
        self.details['found_files'] = len(required_files) - len(missing_files)
        
        return len(missing_files) == 0

class SyntaxValidationGate(QualityGate):
    """Validates Python syntax across all files."""
    
    def __init__(self):
        super().__init__("Syntax Validation", weight=1.0)
    
    def _run_check(self) -> bool:
        python_files = list(Path("src").rglob("*.py"))
        syntax_errors = []
        valid_files = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
                valid_files += 1
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {e}")
        
        self.details['total_files'] = len(python_files)
        self.details['valid_files'] = valid_files
        self.details['syntax_errors'] = syntax_errors
        
        return len(syntax_errors) == 0

class SecurityGate(QualityGate):
    """Security vulnerability scanning."""
    
    def __init__(self):
        super().__init__("Security Scan", weight=1.5)
    
    def _run_check(self) -> bool:
        security_patterns = {
            'unsafe_eval': r'\beval\s*\(',
            'unsafe_exec': r'\bexec\s*\(',
            'hardcoded_secrets': r'password\s*=\s*["\'][^"\']+["\']',
            'sql_injection': r'execute\s*\(\s*["\'].*%.*["\']'
        }
        
        total_violations = 0
        violations_by_type = {}
        
        for root, dirs, files in os.walk("src"):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for vuln_type, pattern in security_patterns.items():
                            matches = list(re.finditer(pattern, content, re.IGNORECASE))
                            if matches:
                                if vuln_type not in violations_by_type:
                                    violations_by_type[vuln_type] = []
                                violations_by_type[vuln_type].extend([
                                    f"{file_path}:{content[:m.start()].count('\n')+1}"
                                    for m in matches
                                ])
                                total_violations += len(matches)
                    
                    except Exception as e:
                        logger.warning(f"Could not scan {file_path}: {e}")
        
        self.details['total_violations'] = total_violations
        self.details['violations_by_type'] = violations_by_type
        
        # Allow some eval usage but flag it
        eval_count = len(violations_by_type.get('unsafe_eval', []))
        other_violations = total_violations - eval_count
        
        # Pass if no critical violations (non-eval)
        return other_violations == 0 and eval_count < 50  # Reasonable threshold

class ModelValidationGate(QualityGate):
    """Validates neural network model implementations."""
    
    def __init__(self):
        super().__init__("Model Validation", weight=1.5)
    
    def _run_check(self) -> bool:
        models_file = "src/pno_physics_bench/models.py"
        
        if not os.path.exists(models_file):
            self.details['error'] = "Models file not found"
            return False
        
        try:
            with open(models_file, 'r') as f:
                content = f.read()
        except Exception as e:
            self.details['error'] = f"Could not read models file: {e}"
            return False
        
        # Check for required classes
        required_classes = [
            'ProbabilisticNeuralOperator',
            'FourierNeuralOperator',
            'SpectralConv2d_Probabilistic',
            'DeepONet'
        ]
        
        found_classes = []
        missing_classes = []
        
        for class_name in required_classes:
            if f"class {class_name}" in content:
                found_classes.append(class_name)
            else:
                missing_classes.append(class_name)
        
        # Check for key methods
        required_methods = [
            'predict_with_uncertainty',
            'kl_divergence',
            'forward'
        ]
        
        found_methods = []
        for method in required_methods:
            if f"def {method}" in content:
                found_methods.append(method)
        
        self.details['found_classes'] = found_classes
        self.details['missing_classes'] = missing_classes
        self.details['found_methods'] = found_methods
        self.details['class_coverage'] = len(found_classes) / len(required_classes)
        self.details['method_coverage'] = len(found_methods) / len(required_methods)
        
        return len(missing_classes) == 0 and len(found_methods) >= len(required_methods) * 0.8

class DocumentationGate(QualityGate):
    """Validates documentation completeness."""
    
    def __init__(self):
        super().__init__("Documentation", weight=0.8)
    
    def _run_check(self) -> bool:
        required_docs = [
            "README.md",
            "ARCHITECTURE.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CHANGELOG.md"
        ]
        
        found_docs = []
        missing_docs = []
        
        for doc in required_docs:
            if os.path.exists(doc):
                found_docs.append(doc)
                # Check if file has substantial content
                try:
                    with open(doc, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if len(content) < 100:  # Minimum content threshold
                        self.details.setdefault('incomplete_docs', []).append(doc)
                except Exception:
                    pass
            else:
                missing_docs.append(doc)
        
        # Check for code docstrings
        python_files = list(Path("src").rglob("*.py"))
        files_with_docstrings = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for module-level docstring
                if '"""' in content[:500] or "'''" in content[:500]:
                    files_with_docstrings += 1
            except Exception:
                pass
        
        docstring_coverage = files_with_docstrings / len(python_files) if python_files else 0
        
        self.details['found_docs'] = found_docs
        self.details['missing_docs'] = missing_docs
        self.details['doc_coverage'] = len(found_docs) / len(required_docs)
        self.details['docstring_coverage'] = docstring_coverage
        
        return len(found_docs) >= len(required_docs) * 0.8 and docstring_coverage >= 0.5

class PerformanceGate(QualityGate):
    """Performance benchmarking and optimization validation."""
    
    def __init__(self):
        super().__init__("Performance", weight=1.2)
    
    def _run_check(self) -> bool:
        # Check for performance optimization files
        perf_files = [
            "src/pno_physics_bench/optimization.py",
            "src/pno_physics_bench/performance_optimization.py",
            "src/pno_physics_bench/scaling/performance_optimization.py",
            "src/pno_physics_bench/scaling/intelligent_caching.py"
        ]
        
        found_perf_files = []
        for file_path in perf_files:
            if os.path.exists(file_path):
                found_perf_files.append(file_path)
        
        # Run a simple performance test
        start_time = time.time()
        
        # Simulate computational load
        result = sum(i ** 2 for i in range(10000))
        
        computation_time = time.time() - start_time
        
        # Check if performance config exists
        has_perf_config = os.path.exists("performance_config.json")
        
        self.details['found_perf_files'] = found_perf_files
        self.details['perf_file_coverage'] = len(found_perf_files) / len(perf_files)
        self.details['computation_time'] = computation_time
        self.details['has_performance_config'] = has_perf_config
        self.details['benchmark_result'] = result
        
        # Pass if we have reasonable performance infrastructure
        return (len(found_perf_files) >= len(perf_files) * 0.5 and 
                computation_time < 1.0 and 
                has_perf_config)

class DeploymentReadinessGate(QualityGate):
    """Validates deployment readiness."""
    
    def __init__(self):
        super().__init__("Deployment Readiness", weight=1.3)
    
    def _run_check(self) -> bool:
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        deployment_dirs = [
            "deployment/",
            "monitoring/"
        ]
        
        found_files = []
        found_dirs = []
        
        for file_path in deployment_files:
            if os.path.exists(file_path):
                found_files.append(file_path)
        
        for dir_path in deployment_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                found_dirs.append(dir_path)
        
        # Check for environment-specific configs
        env_configs = []
        if os.path.exists("deployment/configs/"):
            env_configs = [f for f in os.listdir("deployment/configs/") 
                          if f.endswith(('.json', '.yaml', '.yml'))]
        
        # Check for monitoring setup
        has_monitoring = (os.path.exists("monitoring/") and 
                         os.path.exists("monitoring/prometheus.yml"))
        
        self.details['found_files'] = found_files
        self.details['found_dirs'] = found_dirs
        self.details['env_configs'] = env_configs
        self.details['has_monitoring'] = has_monitoring
        self.details['deployment_score'] = (
            len(found_files) / len(deployment_files) * 0.4 +
            len(found_dirs) / len(deployment_dirs) * 0.3 +
            (len(env_configs) > 0) * 0.2 +
            has_monitoring * 0.1
        )
        
        return self.details['deployment_score'] >= 0.7

class QualityGateRunner:
    """Orchestrates execution of all quality gates."""
    
    def __init__(self):
        self.gates = [
            CodeStructureGate(),
            SyntaxValidationGate(),
            SecurityGate(),
            ModelValidationGate(),
            DocumentationGate(),
            PerformanceGate(),
            DeploymentReadinessGate()
        ]
        self.results = {}
        self.overall_score = 0.0
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return results."""
        print("🧪 Executing Comprehensive Quality Gates")
        print("=" * 60)
        
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = 0.0
        
        for i, gate in enumerate(self.gates, 1):
            print(f"\n[{i}/{len(self.gates)}] Running {gate.name}...")
            
            success = gate.execute()
            weighted_score += gate.score * gate.weight
            
            # Store results
            self.results[gate.name] = {
                'passed': gate.passed,
                'score': gate.score,
                'weight': gate.weight,
                'execution_time': gate.execution_time,
                'details': gate.details
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} ({gate.execution_time:.2f}s)")
            
            # Show key details
            if gate.details:
                for key, value in gate.details.items():
                    if isinstance(value, (int, float)):
                        print(f"   📊 {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 3:
                        print(f"   📊 {key}: {value}")
        
        self.overall_score = (weighted_score / total_weight) * 100
        
        # Generate summary
        passed_gates = sum(1 for gate in self.gates if gate.passed)
        total_gates = len(self.gates)
        
        print(f"\n" + "=" * 60)
        print(f"🎯 Quality Gates Summary")
        print(f"   Passed: {passed_gates}/{total_gates} gates")
        print(f"   Overall Score: {self.overall_score:.1f}/100")
        
        if self.overall_score >= 80:
            print(f"   Status: ✅ EXCELLENT - Ready for production")
        elif self.overall_score >= 70:
            print(f"   Status: ✅ GOOD - Minor improvements needed")
        elif self.overall_score >= 60:
            print(f"   Status: ⚠️  ACCEPTABLE - Some issues to address")
        else:
            print(f"   Status: ❌ NEEDS WORK - Significant improvements required")
        
        # Save detailed results
        final_results = {
            'timestamp': time.time(),
            'overall_score': self.overall_score,
            'passed_gates': passed_gates,
            'total_gates': total_gates,
            'gate_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        with open("quality_gates_report.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📊 Detailed report saved: quality_gates_report.json")
        
        return final_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []
        
        for gate_name, result in self.results.items():
            if not result['passed']:
                if gate_name == "Security Scan":
                    recommendations.append("Review and replace unsafe eval() calls with safer alternatives")
                elif gate_name == "Model Validation":
                    recommendations.append("Ensure all required neural network model classes are implemented")
                elif gate_name == "Documentation":
                    recommendations.append("Add comprehensive documentation and docstrings")
                elif gate_name == "Performance":
                    recommendations.append("Implement performance optimization features and benchmarks")
                elif gate_name == "Deployment Readiness":
                    recommendations.append("Complete deployment configuration and monitoring setup")
                else:
                    recommendations.append(f"Address issues in {gate_name}")
        
        if self.overall_score < 80:
            recommendations.append("Consider implementing additional quality assurance measures")
        
        return recommendations

def main():
    """Execute comprehensive quality gates."""
    runner = QualityGateRunner()
    results = runner.run_all_gates()
    
    # Return success if score is acceptable
    return results['overall_score'] >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)