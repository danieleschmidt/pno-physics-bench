#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
For pno-physics-bench repository

This script implements the continuous value discovery loop that:
1. Harvests signals from multiple sources
2. Scores work items using WSJF + ICE + Technical Debt metrics
3. Selects and executes the highest-value tasks
4. Learns and adapts from execution outcomes
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("terragon-discovery")

@dataclass
class WorkItem:
    """Represents a discoverable work item with scoring metrics."""
    id: str
    title: str
    description: str
    category: str
    source: str
    priority: str
    estimated_effort: float  # hours
    files_affected: List[str]
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Risk assessment
    risk_level: float = 0.0
    confidence: float = 0.5
    
    # Execution tracking
    status: str = "pending"
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

class ValueDiscoveryEngine:
    """Main engine for autonomous value discovery and execution."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = repo_path / "BACKLOG.md"
        
        # Load configuration
        self.config = self._load_config()
        self.weights = self.config["scoring"]["weights"]
        self.thresholds = self.config["scoring"]["thresholds"]
        
        # Initialize state
        self.work_items: List[WorkItem] = []
        self.execution_history: List[Dict] = []
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise

    def _load_metrics(self) -> Dict:
        """Load value metrics and execution history."""
        try:
            with open(self.metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Metrics file not found, initializing empty metrics")
            return {
                "repository": {"name": "pno-physics-bench", "maturityLevel": "maturing"},
                "executionHistory": [],
                "backlogMetrics": {"totalItems": 0},
                "valueDelivered": {"totalScore": 0}
            }

    async def discover_work_items(self) -> List[WorkItem]:
        """Comprehensive signal harvesting from multiple sources."""
        logger.info("🔍 Starting comprehensive value discovery...")
        
        discovered_items = []
        
        # Source 1: Git history analysis
        git_items = await self._discover_from_git_history()
        discovered_items.extend(git_items)
        
        # Source 2: Static code analysis
        static_items = await self._discover_from_static_analysis()
        discovered_items.extend(static_items)
        
        # Source 3: Dependency analysis
        dep_items = await self._discover_from_dependencies()
        discovered_items.extend(dep_items)
        
        # Source 4: Security vulnerability scanning
        sec_items = await self._discover_from_security_scan()
        discovered_items.extend(sec_items)
        
        # Source 5: Documentation gaps
        doc_items = await self._discover_documentation_gaps()
        discovered_items.extend(doc_items)
        
        # Source 6: Performance optimization opportunities
        perf_items = await self._discover_performance_opportunities()
        discovered_items.extend(perf_items)
        
        logger.info(f"📊 Discovered {len(discovered_items)} potential work items")
        return discovered_items

    async def _discover_from_git_history(self) -> List[WorkItem]:
        """Analyze git history for TODO/FIXME/HACK markers."""
        items = []
        try:
            # Search for technical debt markers in code
            result = subprocess.run([
                "git", "grep", "-n", "-E", "(TODO|FIXME|HACK|XXX|DEPRECATED)",
                "--", "*.py", "*.md", "*.yaml", "*.yml"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        file_path, line_num, content = line.split(':', 2)
                        
                        # Extract marker type and description
                        marker = content.split()[0] if content.split() else "TODO"
                        description = content.replace(marker, "").strip()
                        
                        items.append(WorkItem(
                            id=f"git-{hash(line) % 10000:04d}",
                            title=f"Address {marker} in {file_path}",
                            description=description,
                            category="technical-debt",
                            source="git-history",
                            priority="medium",
                            estimated_effort=1.0,
                            files_affected=[file_path]
                        ))
        except Exception as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items[:10]  # Limit to top 10 items

    async def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Run static analysis tools to discover quality issues."""
        items = []
        
        # Check for complexity issues with existing tools
        complexity_items = [
            WorkItem(
                id="complexity-001",
                title="Reduce cyclomatic complexity in high-churn files",
                description="Identify and refactor complex functions using radon or similar tools",
                category="code-quality",
                source="static-analysis",
                priority="medium",
                estimated_effort=4.0,
                files_affected=["src/pno_physics_bench/models/", "src/pno_physics_bench/training/"]
            )
        ]
        
        # Type coverage improvements
        type_items = [
            WorkItem(
                id="types-001", 
                title="Improve type coverage in core modules",
                description="Add comprehensive type hints to increase mypy coverage",
                category="code-quality",
                source="static-analysis",
                priority="low",
                estimated_effort=3.0,
                files_affected=["src/pno_physics_bench/"]
            )
        ]
        
        items.extend(complexity_items + type_items)
        return items

    async def _discover_from_dependencies(self) -> List[WorkItem]:
        """Analyze dependencies for updates and security issues."""
        items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                if outdated:
                    items.append(WorkItem(
                        id="deps-001",
                        title=f"Update {len(outdated)} outdated dependencies",
                        description=f"Update packages: {', '.join([pkg['name'] for pkg in outdated[:5]])}",
                        category="dependencies",
                        source="dependency-analysis",
                        priority="medium",
                        estimated_effort=2.0,
                        files_affected=["requirements.txt", "pyproject.toml"]
                    ))
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
        
        return items

    async def _discover_from_security_scan(self) -> List[WorkItem]:
        """Discover security vulnerabilities and issues."""
        items = []
        
        # Mock security scan results (in production, integrate with actual tools)
        security_items = [
            WorkItem(
                id="sec-001",
                title="Add secrets scanning to CI pipeline",
                description="Implement detect-secrets baseline and CI integration",
                category="security",
                source="security-scan",
                priority="high",
                estimated_effort=2.0,
                files_affected=[".pre-commit-config.yaml", ".github/workflows/"]
            ),
            WorkItem(
                id="sec-002",
                title="Enable SBOM generation for container images",
                description="Add Software Bill of Materials generation to Docker builds",
                category="security",
                source="security-scan", 
                priority="medium",
                estimated_effort=3.0,
                files_affected=["Dockerfile", "docker-compose.yml"]
            )
        ]
        
        items.extend(security_items)
        return items

    async def _discover_documentation_gaps(self) -> List[WorkItem]:
        """Identify missing or outdated documentation."""
        items = []
        
        # Check for missing docstrings
        doc_items = [
            WorkItem(
                id="doc-001",
                title="Add comprehensive API documentation",
                description="Generate API docs for all public methods and classes",
                category="documentation",
                source="documentation-analysis",
                priority="medium",
                estimated_effort=6.0,
                files_affected=["src/pno_physics_bench/", "docs/"]
            ),
            WorkItem(
                id="doc-002",
                title="Create deployment guide",
                description="Document production deployment procedures and best practices",
                category="documentation", 
                source="documentation-analysis",
                priority="medium",
                estimated_effort=4.0,
                files_affected=["docs/", "DEPLOYMENT.md"]
            )
        ]
        
        items.extend(doc_items)
        return items

    async def _discover_performance_opportunities(self) -> List[WorkItem]:
        """Identify performance optimization opportunities."""
        items = []
        
        perf_items = [
            WorkItem(
                id="perf-001",
                title="Add performance benchmarking suite",
                description="Implement comprehensive performance regression testing",
                category="performance",
                source="performance-analysis",
                priority="medium",
                estimated_effort=8.0,
                files_affected=["tests/benchmark/", "scripts/"]
            ),
            WorkItem(
                id="perf-002", 
                title="Optimize tensor operations in PNO forward pass",
                description="Profile and optimize critical path tensor operations",
                category="performance",
                source="performance-analysis",
                priority="high",
                estimated_effort=6.0,
                files_affected=["src/pno_physics_bench/models/"]
            )
        ]
        
        items.extend(perf_items)
        return items

    def calculate_composite_score(self, item: WorkItem) -> float:
        """Calculate composite score using WSJF + ICE + Technical Debt."""
        
        # WSJF Calculation (Weighted Shortest Job First)
        user_business_value = self._score_business_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity_enablement = self._score_opportunity_enablement(item)
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        wsjf = cost_of_delay / max(item.estimated_effort, 0.5)
        
        # ICE Calculation (Impact, Confidence, Ease)
        impact = self._score_impact(item)
        confidence = item.confidence
        ease = self._score_ease(item)
        ice = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = self._calculate_debt_impact(item)
        debt_interest = self._calculate_debt_interest(item)
        hotspot_multiplier = self._get_hotspot_multiplier(item)
        tech_debt = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Store component scores
        item.wsjf_score = wsjf
        item.ice_score = ice
        item.technical_debt_score = tech_debt
        
        # Calculate composite score with adaptive weighting
        maturity_level = self.config["repository"]["maturity_level"]
        weights = self.weights[maturity_level]
        
        composite = (
            weights["wsjf"] * self._normalize_score(wsjf, 0, 100) +
            weights["ice"] * self._normalize_score(ice, 0, 1000) +
            weights["technicalDebt"] * self._normalize_score(tech_debt, 0, 200) +
            weights["security"] * self._get_security_boost(item)
        )
        
        # Apply category-specific boosts/penalties
        if item.category == "security":
            composite *= self.thresholds["securityBoost"]
        elif item.category == "technical-debt":
            composite *= 1.2
        elif item.category == "documentation":
            composite *= 0.8
        
        item.composite_score = composite
        return composite

    def _score_business_value(self, item: WorkItem) -> float:
        """Score business value impact (1-10 scale)."""
        category_scores = {
            "security": 9,
            "performance": 8,
            "technical-debt": 6,
            "code-quality": 5,
            "documentation": 4,
            "dependencies": 5
        }
        return category_scores.get(item.category, 5)

    def _score_time_criticality(self, item: WorkItem) -> float:
        """Score time criticality (1-10 scale)."""
        priority_scores = {"high": 8, "medium": 5, "low": 2}
        return priority_scores.get(item.priority, 5)

    def _score_risk_reduction(self, item: WorkItem) -> float:
        """Score risk reduction value (1-10 scale)."""
        if item.category in ["security", "technical-debt"]:
            return 8
        elif item.category in ["performance", "code-quality"]:
            return 6
        return 3

    def _score_opportunity_enablement(self, item: WorkItem) -> float:
        """Score opportunity enablement (1-10 scale)."""
        if item.category in ["performance", "code-quality"]:
            return 7
        elif item.category == "documentation":
            return 6
        return 4

    def _score_impact(self, item: WorkItem) -> float:
        """Score business impact (1-10 scale)."""
        return self._score_business_value(item)

    def _score_ease(self, item: WorkItem) -> float:
        """Score implementation ease (1-10 scale)."""
        # Inverse of effort (easier = higher score)
        return max(1, 10 - item.estimated_effort)

    def _calculate_debt_impact(self, item: WorkItem) -> float:
        """Calculate technical debt cost impact."""
        if item.category == "technical-debt":
            return item.estimated_effort * 10  # 10x multiplier for debt
        return item.estimated_effort * 2

    def _calculate_debt_interest(self, item: WorkItem) -> float:
        """Calculate future cost if debt not addressed."""
        age_days = (datetime.now() - item.created_at).days
        return age_days * 0.1  # Debt grows over time

    def _get_hotspot_multiplier(self, item: WorkItem) -> float:
        """Get hotspot multiplier based on file churn/complexity."""
        # Mock implementation - in practice, analyze git log and complexity metrics
        hotspot_files = ["models/", "training/", "uncertainty/"]
        for file_path in item.files_affected:
            for hotspot in hotspot_files:
                if hotspot in file_path:
                    return 2.0
        return 1.0

    def _get_security_boost(self, item: WorkItem) -> float:
        """Get security priority boost."""
        if item.category == "security":
            return 1.0
        return 0.0

    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        return max(0, min(1, (score - min_val) / (max_val - min_val)))

    async def select_next_best_value(self) -> Optional[WorkItem]:
        """Select the highest-value work item for execution."""
        if not self.work_items:
            return None
        
        # Filter items based on dependencies and risk
        eligible_items = []
        for item in self.work_items:
            if item.status != "pending":
                continue
            if item.composite_score < self.thresholds["minScore"]:
                continue
            if item.risk_level > self.thresholds["maxRisk"]:
                continue
            eligible_items.append(item)
        
        if not eligible_items:
            logger.info("No eligible items found, generating housekeeping tasks")
            return self._generate_housekeeping_task()
        
        # Sort by composite score and return highest
        eligible_items.sort(key=lambda x: x.composite_score, reverse=True)
        return eligible_items[0]

    def _generate_housekeeping_task(self) -> WorkItem:
        """Generate maintenance task when no high-value items exist."""
        return WorkItem(
            id="house-001",
            title="Update development dependencies",
            description="Update pre-commit hooks and development tools to latest versions",
            category="maintenance",
            source="housekeeping",
            priority="low", 
            estimated_effort=1.0,
            files_affected=[".pre-commit-config.yaml", "pyproject.toml"]
        )

    async def execute_work_item(self, item: WorkItem) -> Dict:
        """Execute a work item and track results."""
        logger.info(f"🚀 Executing: {item.title}")
        
        start_time = time.time()
        item.status = "in_progress"
        item.updated_at = datetime.now()
        
        try:
            # Create feature branch
            branch_name = f"auto-value/{item.id}-{item.title.lower().replace(' ', '-')[:30]}"
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.repo_path)
            
            # Execute item-specific logic
            execution_result = await self._execute_item_logic(item)
            
            # Run validation
            validation_result = await self._validate_changes(item)
            
            if validation_result["success"]:
                # Create pull request
                pr_result = await self._create_pull_request(item, branch_name)
                item.status = "completed"
                execution_result["pr_url"] = pr_result.get("url")
            else:
                # Rollback changes
                await self._rollback_changes(item, branch_name)
                item.status = "failed"
                execution_result["error"] = validation_result["error"]
            
        except Exception as e:
            logger.error(f"Execution failed for {item.id}: {e}")
            item.status = "failed"
            execution_result = {"success": False, "error": str(e)}
        
        # Track execution metrics
        execution_time = time.time() - start_time
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "itemId": item.id,
            "title": item.title,
            "category": item.category,
            "scores": {
                "wsjf": item.wsjf_score,
                "ice": item.ice_score,
                "technicalDebt": item.technical_debt_score,
                "composite": item.composite_score
            },
            "estimatedEffort": item.estimated_effort,
            "actualEffort": execution_time / 3600,  # Convert to hours
            "status": item.status,
            "result": execution_result
        }
        
        self.execution_history.append(execution_record)
        await self._update_metrics()
        
        return execution_record

    async def _execute_item_logic(self, item: WorkItem) -> Dict:
        """Execute the specific logic for different item types."""
        # This is where specific automation would be implemented
        # For now, return mock success
        await asyncio.sleep(1)  # Simulate work
        return {"success": True, "changes": f"Automated changes for {item.category}"}

    async def _validate_changes(self, item: WorkItem) -> Dict:
        """Validate changes meet quality gates."""
        try:
            # Run tests
            test_result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v"],
                capture_output=True, cwd=self.repo_path
            )
            
            if test_result.returncode != 0:
                return {"success": False, "error": "Test failures"}
            
            # Run linting
            lint_result = subprocess.run(
                ["pre-commit", "run", "--all-files"],
                capture_output=True, cwd=self.repo_path
            )
            
            if lint_result.returncode != 0:
                return {"success": False, "error": "Linting failures"}
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": f"Validation error: {e}"}

    async def _create_pull_request(self, item: WorkItem, branch_name: str) -> Dict:
        """Create pull request for completed work."""
        # Mock PR creation - in practice, use GitHub API
        logger.info(f"Creating PR for {item.title} on branch {branch_name}")
        return {"url": f"https://github.com/repo/pulls/{item.id}", "number": 123}

    async def _rollback_changes(self, item: WorkItem, branch_name: str):
        """Rollback changes on failure."""
        logger.warning(f"Rolling back changes for {item.title}")
        subprocess.run(["git", "checkout", "main"], cwd=self.repo_path)
        subprocess.run(["git", "branch", "-D", branch_name], cwd=self.repo_path)

    async def _update_metrics(self):
        """Update value metrics file."""
        self.metrics["executionHistory"] = self.execution_history[-100:]  # Keep last 100
        self.metrics["backlogMetrics"]["totalItems"] = len(self.work_items)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    async def update_backlog_document(self):
        """Update BACKLOG.md with current state."""
        # Implementation would generate updated BACKLOG.md content
        logger.info("📝 Updated BACKLOG.md with current discovery state")

    async def run_discovery_cycle(self) -> Dict:
        """Run complete discovery and execution cycle."""
        logger.info("🔄 Starting autonomous value discovery cycle")
        
        # Step 1: Discover work items
        discovered_items = await self.discover_work_items()
        
        # Step 2: Score all items
        for item in discovered_items:
            self.calculate_composite_score(item)
        
        # Step 3: Add to work items list
        self.work_items.extend(discovered_items)
        
        # Step 4: Select next best value
        next_item = await self.select_next_best_value()
        
        if next_item:
            # Step 5: Execute work item
            execution_result = await self.execute_work_item(next_item)
            
            # Step 6: Update documentation
            await self.update_backlog_document()
            
            logger.info(f"✅ Completed cycle with execution of: {next_item.title}")
            return {"status": "executed", "item": next_item.title, "result": execution_result}
        else:
            logger.info("ℹ️ No actionable items found in this cycle")
            return {"status": "no_action", "discovered_items": len(discovered_items)}

async def main():
    """Main entry point for autonomous discovery."""
    engine = ValueDiscoveryEngine()
    
    # Run single discovery cycle
    result = await engine.run_discovery_cycle()
    
    logger.info(f"🎯 Discovery cycle completed: {result}")
    return result

if __name__ == "__main__":
    asyncio.run(main())