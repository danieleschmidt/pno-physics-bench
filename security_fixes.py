#!/usr/bin/env python3
"""
Security Enhancement Script
Replaces unsafe eval() calls with safer alternatives
"""

import os
import re
import ast
import json
from typing import Dict, List, Tuple

def create_safe_evaluator():
    """Create a safe expression evaluator."""
    return """
import ast
import operator
from typing import Any, Dict

class SafeEvaluator:
    \"\"\"Safe expression evaluator that only allows mathematical operations.\"\"\"
    
    # Allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    @classmethod
    def safe_eval(cls, expr: str, variables: Dict[str, Any] = None) -> Any:
        \"\"\"Safely evaluate mathematical expressions.\"\"\"
        if variables is None:
            variables = {}
            
        try:
            node = ast.parse(expr, mode='eval')
            return cls._eval_node(node.body, variables)
        except Exception as e:
            raise ValueError(f"Invalid expression: {expr}") from e
    
    @classmethod
    def _eval_node(cls, node: ast.AST, variables: Dict[str, Any]) -> Any:
        \"\"\"Recursively evaluate AST nodes.\"\"\"
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            else:
                raise NameError(f"Variable '{node.id}' not defined")
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left, variables)
            right = cls._eval_node(node.right, variables) 
            op = cls.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand, variables)
            op = cls.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
"""

def fix_eval_usage(file_path: str) -> bool:
    """Fix unsafe eval() usage in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to find eval() calls
        eval_pattern = r'\beval\s*\(\s*([^)]+)\s*\)'
        
        def replace_eval(match):
            expr = match.group(1).strip()
            # For simple cases, try to use safer alternatives
            if expr.startswith('"') or expr.startswith("'"):
                # String literal - use ast.literal_eval
                return f"ast.literal_eval({expr})"
            else:
                # Variable or complex expression - use SafeEvaluator
                return f"SafeEvaluator.safe_eval({expr})"
        
        # Replace eval calls
        new_content = re.sub(eval_pattern, replace_eval, content)
        
        # Add imports if eval was replaced
        if new_content != original_content:
            if 'import ast' not in new_content:
                import_line = "import ast\n"
                if new_content.startswith('"""') or new_content.startswith("'''"):
                    # Insert after docstring
                    docstring_end = new_content.find('"""', 3)
                    if docstring_end == -1:
                        docstring_end = new_content.find("'''", 3)
                    if docstring_end != -1:
                        docstring_end += 3
                        new_content = new_content[:docstring_end] + "\n\n" + import_line + new_content[docstring_end:]
                    else:
                        new_content = import_line + new_content
                else:
                    new_content = import_line + new_content
            
            # Add SafeEvaluator class if needed
            if 'SafeEvaluator.safe_eval' in new_content and 'class SafeEvaluator:' not in new_content:
                safe_eval_code = create_safe_evaluator()
                # Insert after imports
                lines = new_content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                        insert_pos = i
                        break
                
                lines.insert(insert_pos, safe_eval_code)
                new_content = '\n'.join(lines)
        
        # Write back if changed
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_security_issues():
    """Fix security issues across the codebase."""
    print("🔒 Applying Security Fixes")
    print("-" * 50)
    
    # Files with eval() usage that need fixing
    problematic_files = [
        "src/pno_physics_bench/models.py",
        "src/pno_physics_bench/uncertainty.py", 
        "src/pno_physics_bench/training/trainer.py"
    ]
    
    fixed_files = 0
    for file_path in problematic_files:
        if os.path.exists(file_path):
            print(f"🔧 Fixing {file_path}...")
            if fix_eval_usage(file_path):
                print(f"✅ Fixed unsafe eval() in {file_path}")
                fixed_files += 1
            else:
                print(f"ℹ️  No changes needed in {file_path}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    return fixed_files

def create_security_config():
    """Create security configuration file."""
    security_config = {
        "security_policies": {
            "code_scanning": {
                "enabled": True,
                "scan_patterns": [
                    "eval\\s*\\(",
                    "exec\\s*\\(",
                    "password\\s*=",
                    "api_key\\s*="
                ],
                "excluded_files": [
                    "tests/*",
                    "*.md",
                    "*.txt"
                ]
            },
            "input_validation": {
                "enabled": True,
                "max_input_size": 10000,
                "allowed_file_types": [".py", ".yaml", ".json"],
                "sanitize_paths": True
            },
            "access_control": {
                "enforce_permissions": True,
                "readonly_directories": ["docs/", "examples/"],
                "restricted_imports": ["os.system", "subprocess.call"]
            }
        },
        "monitoring": {
            "log_security_events": True,
            "alert_on_violations": True,
            "audit_trail": True
        }
    }
    
    with open("security_config.json", "w") as f:
        json.dump(security_config, f, indent=2)
    
    print("✅ Security configuration created: security_config.json")

def main():
    """Apply security enhancements."""
    print("🛡️ Security Enhancement Suite")
    print("=" * 50)
    
    # Fix eval() usage
    fixed_count = fix_security_issues()
    
    # Create security configuration
    create_security_config()
    
    # Summary
    print(f"\n📊 Security Enhancement Summary:")
    print(f"   Fixed files: {fixed_count}")
    print(f"   Security config: Created")
    
    if fixed_count > 0:
        print("\n✅ Security fixes applied successfully!")
        print("⚠️  Note: Review the changes and test functionality")
    else:
        print("\nℹ️  No security fixes needed")
    
    return True

if __name__ == "__main__":
    main()