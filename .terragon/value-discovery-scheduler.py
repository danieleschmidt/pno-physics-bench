#!/usr/bin/env python3
"""
Terragon Value Discovery Scheduler
Implements continuous scheduling and execution of value discovery cycles

This scheduler manages:
- Trigger-based execution (PR merge, issue creation, etc.)
- Time-based execution (hourly, daily, weekly cycles)
- Priority-based execution queue
- Resource management and throttling
- Failure recovery and retry logic
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from autonomous_discovery import ValueDiscoveryEngine, WorkItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("terragon-scheduler")

class ValueDiscoveryScheduler:
    """Continuous scheduler for autonomous value discovery."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.state_path = repo_path / ".terragon" / "scheduler-state.json"
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize discovery engine
        self.discovery_engine = ValueDiscoveryEngine(repo_path)
        
        # Scheduler state
        self.running = False
        self.execution_queue: List[Dict] = []
        self.last_executions: Dict[str, datetime] = {}
        
        # Load persisted state
        self._load_state()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
    
    def _load_state(self):
        """Load scheduler state from disk."""
        try:
            if self.state_path.exists():
                with open(self.state_path) as f:
                    state = json.load(f)
                    self.last_executions = {
                        k: datetime.fromisoformat(v) 
                        for k, v in state.get("last_executions", {}).items()
                    }
                    self.execution_queue = state.get("execution_queue", [])
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    def _save_state(self):
        """Save scheduler state to disk."""
        try:
            state = {
                "last_executions": {
                    k: v.isoformat() 
                    for k, v in self.last_executions.items()
                },
                "execution_queue": self.execution_queue
            }
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self._save_state()
        sys.exit(0)
    
    async def schedule_discovery(self, trigger_type: str, priority: str = "medium", 
                               metadata: Optional[Dict] = None):
        """Schedule a value discovery execution."""
        execution_item = {
            "id": f"{trigger_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "trigger_type": trigger_type,
            "priority": priority,
            "scheduled_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "queued"
        }
        
        # Add to queue based on priority
        if priority == "high":
            self.execution_queue.insert(0, execution_item)
        else:
            self.execution_queue.append(execution_item)
        
        logger.info(f"📅 Scheduled {trigger_type} discovery with {priority} priority")
        self._save_state()
    
    async def should_execute_scheduled_discovery(self, schedule_type: str) -> bool:
        """Check if scheduled discovery should run based on timing."""
        now = datetime.now()
        last_execution = self.last_executions.get(schedule_type)
        
        intervals = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1), 
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }
        
        if schedule_type not in intervals:
            return False
        
        if last_execution is None:
            return True
        
        return now - last_execution >= intervals[schedule_type]
    
    async def execute_next_discovery(self) -> Optional[Dict]:
        """Execute the next discovery from the queue."""
        if not self.execution_queue:
            return None
        
        # Get highest priority item
        execution_item = self.execution_queue.pop(0)
        execution_item["status"] = "executing"
        execution_item["started_at"] = datetime.now().isoformat()
        
        logger.info(f"🚀 Executing discovery: {execution_item['id']}")
        
        try:
            # Run discovery cycle
            result = await self.discovery_engine.run_discovery_cycle()
            
            # Update execution record
            execution_item["status"] = "completed"
            execution_item["completed_at"] = datetime.now().isoformat()
            execution_item["result"] = result
            
            # Update last execution tracking
            self.last_executions[execution_item["trigger_type"]] = datetime.now()
            
            logger.info(f"✅ Discovery completed: {execution_item['id']}")
            
        except Exception as e:
            logger.error(f"❌ Discovery failed: {execution_item['id']} - {e}")
            execution_item["status"] = "failed"
            execution_item["error"] = str(e)
            execution_item["failed_at"] = datetime.now().isoformat()
            
            # Reschedule with lower priority if not critical
            if execution_item["priority"] != "critical":
                await self.schedule_discovery(
                    trigger_type=f"retry-{execution_item['trigger_type']}",
                    priority="low",
                    metadata={"original_id": execution_item["id"], "retry_count": 1}
                )
        
        self._save_state()
        return execution_item
    
    async def handle_github_webhook(self, event_type: str, payload: Dict):
        """Handle GitHub webhook events for reactive discovery."""
        logger.info(f"🔗 Received GitHub webhook: {event_type}")
        
        # PR merged - high priority discovery
        if event_type == "pull_request" and payload.get("action") == "closed" and payload.get("merged"):
            await self.schedule_discovery(
                trigger_type="pr_merged",
                priority="high",
                metadata={
                    "pr_number": payload["pull_request"]["number"],
                    "pr_title": payload["pull_request"]["title"],
                    "files_changed": len(payload.get("pull_request", {}).get("changed_files", []))
                }
            )
        
        # Issue created - medium priority
        elif event_type == "issues" and payload.get("action") == "opened":
            await self.schedule_discovery(
                trigger_type="issue_created",
                priority="medium",
                metadata={
                    "issue_number": payload["issue"]["number"],
                    "issue_title": payload["issue"]["title"],
                    "labels": [label["name"] for label in payload["issue"]["labels"]]
                }
            )
        
        # Release created - low priority (documentation updates)
        elif event_type == "release" and payload.get("action") == "published":
            await self.schedule_discovery(
                trigger_type="release_published",
                priority="low",
                metadata={
                    "release_tag": payload["release"]["tag_name"],
                    "release_name": payload["release"]["name"]
                }
            )
    
    async def run_scheduled_discoveries(self):
        """Check and run scheduled discoveries."""
        schedule_types = ["hourly", "daily", "weekly", "monthly"]
        
        for schedule_type in schedule_types:
            if await self.should_execute_scheduled_discovery(schedule_type):
                await self.schedule_discovery(
                    trigger_type=f"scheduled_{schedule_type}",
                    priority="medium" if schedule_type in ["hourly", "daily"] else "low"
                )
    
    async def health_check(self) -> Dict:
        """Perform health check of the discovery system."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "queue_length": len(self.execution_queue),
            "last_executions": {
                k: v.isoformat() for k, v in self.last_executions.items()
            },
            "discovery_engine_status": "operational"
        }
        
        # Check if discovery engine is responsive
        try:
            test_items = await self.discovery_engine.discover_work_items()
            health_status["last_discovery_count"] = len(test_items)
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["discovery_engine_status"] = f"error: {e}"
        
        # Check queue age
        if self.execution_queue:
            oldest_item = min(
                self.execution_queue,
                key=lambda x: datetime.fromisoformat(x["scheduled_at"])
            )
            queue_age = datetime.now() - datetime.fromisoformat(oldest_item["scheduled_at"])
            if queue_age > timedelta(hours=24):
                health_status["status"] = "degraded"
                health_status["queue_warning"] = "Items in queue for >24 hours"
        
        return health_status
    
    async def cleanup_old_executions(self):
        """Clean up old execution records and state."""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Clean up last_executions older than 30 days
        to_remove = [
            k for k, v in self.last_executions.items()
            if v < cutoff_date
        ]
        
        for key in to_remove:
            del self.last_executions[key]
        
        logger.info(f"🧹 Cleaned up {len(to_remove)} old execution records")
        self._save_state()
    
    async def generate_metrics(self) -> Dict:
        """Generate operational metrics for monitoring."""
        now = datetime.now()
        
        # Calculate execution frequency
        recent_executions = {
            k: v for k, v in self.last_executions.items()
            if now - v < timedelta(days=7)
        }
        
        metrics = {
            "scheduler": {
                "queue_length": len(self.execution_queue),
                "executions_last_7_days": len(recent_executions),
                "average_queue_time": self._calculate_average_queue_time(),
                "uptime_hours": (now - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
            },
            "discovery": {
                "last_execution": max(self.last_executions.values()).isoformat() if self.last_executions else None,
                "total_trigger_types": len(set(item["trigger_type"] for item in self.execution_queue)),
                "priority_distribution": self._calculate_priority_distribution()
            }
        }
        
        return metrics
    
    def _calculate_average_queue_time(self) -> float:
        """Calculate average time items spend in queue."""
        if not self.execution_queue:
            return 0.0
        
        now = datetime.now()
        total_time = sum(
            (now - datetime.fromisoformat(item["scheduled_at"])).total_seconds()
            for item in self.execution_queue
        )
        
        return total_time / len(self.execution_queue) / 3600  # Convert to hours
    
    def _calculate_priority_distribution(self) -> Dict[str, int]:
        """Calculate distribution of priorities in queue."""
        distribution = {}
        for item in self.execution_queue:
            priority = item["priority"]
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution
    
    async def start_scheduler(self):
        """Start the continuous scheduler loop."""
        logger.info("🎯 Starting Terragon Value Discovery Scheduler")
        self.running = True
        
        while self.running:
            try:
                # Check for scheduled discoveries
                await self.run_scheduled_discoveries()
                
                # Execute next queued discovery
                if self.execution_queue:
                    result = await self.execute_next_discovery()
                    if result:
                        logger.info(f"📊 Discovery result: {result['status']}")
                
                # Periodic cleanup
                if datetime.now().hour == 3:  # 3 AM cleanup
                    await self.cleanup_old_executions()
                
                # Health check logging
                health = await self.health_check()
                if health["status"] != "healthy":
                    logger.warning(f"⚠️ Health check: {health['status']}")
                
                # Save state periodically
                self._save_state()
                
                # Wait before next cycle (30 seconds)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"💥 Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def stop_scheduler(self):
        """Stop the scheduler gracefully."""
        logger.info("🛑 Stopping Terragon Value Discovery Scheduler")
        self.running = False
        self._save_state()

# CLI Interface
async def main():
    """Main entry point for the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Value Discovery Scheduler")
    parser.add_argument("--command", choices=["start", "status", "health", "metrics", "trigger"], 
                       default="start", help="Command to execute")
    parser.add_argument("--trigger-type", help="Trigger type for manual execution")
    parser.add_argument("--priority", choices=["low", "medium", "high", "critical"], 
                       default="medium", help="Priority for manual trigger")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    scheduler = ValueDiscoveryScheduler()
    
    if args.command == "start":
        if args.daemon:
            # TODO: Implement proper daemonization
            logger.info("🔄 Running in daemon mode")
        await scheduler.start_scheduler()
    
    elif args.command == "status":
        health = await scheduler.health_check()
        print(json.dumps(health, indent=2))
    
    elif args.command == "health":
        health = await scheduler.health_check()
        if health["status"] == "healthy":
            print("✅ Scheduler is healthy")
            sys.exit(0)
        else:
            print(f"❌ Scheduler status: {health['status']}")
            sys.exit(1)
    
    elif args.command == "metrics":
        metrics = await scheduler.generate_metrics()
        print(json.dumps(metrics, indent=2))
    
    elif args.command == "trigger":
        if not args.trigger_type:
            print("❌ --trigger-type required for manual trigger")
            sys.exit(1)
        
        await scheduler.schedule_discovery(
            trigger_type=args.trigger_type,
            priority=args.priority
        )
        print(f"✅ Scheduled {args.trigger_type} discovery with {args.priority} priority")

if __name__ == "__main__":
    asyncio.run(main())