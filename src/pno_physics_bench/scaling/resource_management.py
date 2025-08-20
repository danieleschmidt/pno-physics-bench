# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Auto-Scaling Framework for PNO Physics Bench"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions"""
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    queue_length: int
    response_time: float
    error_rate: float
    timestamp: float

@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions"""
    metric_name: str
    threshold_up: float
    threshold_down: float
    cooldown_period: int  # seconds
    min_instances: int
    max_instances: int
    scale_up_step: int = 1
    scale_down_step: int = 1

class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self, history_size: int = 100):
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = ScalingMetrics(0, 0, 0, 0, 0, 0, time.time())
        self._lock = threading.Lock()
    
    def collect_metrics(self, additional_metrics: Optional[Dict] = None) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Get additional metrics if provided
            additional = additional_metrics or {}
            active_tasks = additional.get('active_tasks', 0)
            queue_length = additional.get('queue_length', 0)
            response_time = additional.get('response_time', 0.0)
            error_rate = additional.get('error_rate', 0.0)
            
            metrics = ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_tasks=active_tasks,
                queue_length=queue_length,
                response_time=response_time,
                error_rate=error_rate,
                timestamp=time.time()
            )
            
            with self._lock:
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return self.current_metrics
    
    def get_average_metrics(self, window_seconds: int = 300) -> Optional[ScalingMetrics]:
        """Get average metrics over time window"""
        with self._lock:
            if not self.metrics_history:
                return None
            
            current_time = time.time()
            recent_metrics = [
                m for m in self.metrics_history
                if current_time - m.timestamp <= window_seconds
            ]
            
            if not recent_metrics:
                return None
            
            avg_metrics = ScalingMetrics(
                cpu_usage=sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                memory_usage=sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                active_tasks=sum(m.active_tasks for m in recent_metrics) / len(recent_metrics),
                queue_length=sum(m.queue_length for m in recent_metrics) / len(recent_metrics),
                response_time=sum(m.response_time for m in recent_metrics) / len(recent_metrics),
                error_rate=sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
                timestamp=current_time
            )
            
            return avg_metrics

class AutoScaler:
    """Intelligent auto-scaling engine"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.scaling_rules = []
        self.current_instances = 1
        self.last_scaling_action = 0
        self.scaling_history = deque(maxlen=50)
        self.callbacks = {
            'scale_up': [],
            'scale_down': []
        }
        self.is_running = False
        self._lock = threading.Lock()
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule"""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule for {rule.metric_name}")
    
    def add_callback(self, action: str, callback: Callable[[int], None]):
        """Add callback for scaling actions"""
        if action in self.callbacks:
            self.callbacks[action].append(callback)
    
    def start_monitoring(self, check_interval: int = 30):
        """Start auto-scaling monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.is_running = False
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self.metrics_collector.collect_metrics()
                
                # Check scaling rules
                scaling_decision = self._evaluate_scaling_rules(current_metrics)
                
                if scaling_decision != 0:
                    self._execute_scaling_action(scaling_decision)
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")
                time.sleep(check_interval)
    
    def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> int:
        """Evaluate scaling rules and return scaling decision"""
        current_time = time.time()
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.scaling_rules:
            # Check cooldown period
            if current_time - self.last_scaling_action < rule.cooldown_period:
                continue
            
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, 0)
            
            # Check thresholds
            if metric_value > rule.threshold_up and self.current_instances < rule.max_instances:
                scale_up_votes += 1
            elif metric_value < rule.threshold_down and self.current_instances > rule.min_instances:
                scale_down_votes += 1
        
        # Determine scaling action
        if scale_up_votes > scale_down_votes:
            return 1  # Scale up
        elif scale_down_votes > scale_up_votes:
            return -1  # Scale down
        else:
            return 0  # No scaling
    
    def _execute_scaling_action(self, decision: int):
        """Execute scaling action"""
        with self._lock:
            current_time = time.time()
            
            if decision > 0:  # Scale up
                old_instances = self.current_instances
                
                # Find the appropriate scaling rule
                scale_step = 1
                for rule in self.scaling_rules:
                    if self.current_instances < rule.max_instances:
                        scale_step = rule.scale_up_step
                        break
                
                self.current_instances = min(
                    self.current_instances + scale_step,
                    max(rule.max_instances for rule in self.scaling_rules)
                )
                
                logger.info(f"Scaling UP: {old_instances} -> {self.current_instances} instances")
                
                # Execute scale-up callbacks
                for callback in self.callbacks['scale_up']:
                    try:
                        callback(self.current_instances - old_instances)
                    except Exception as e:
                        logger.error(f"Scale-up callback error: {e}")
                
            elif decision < 0:  # Scale down
                old_instances = self.current_instances
                
                # Find the appropriate scaling rule
                scale_step = 1
                for rule in self.scaling_rules:
                    if self.current_instances > rule.min_instances:
                        scale_step = rule.scale_down_step
                        break
                
                self.current_instances = max(
                    self.current_instances - scale_step,
                    max(rule.min_instances for rule in self.scaling_rules)
                )
                
                logger.info(f"Scaling DOWN: {old_instances} -> {self.current_instances} instances")
                
                # Execute scale-down callbacks
                for callback in self.callbacks['scale_down']:
                    try:
                        callback(old_instances - self.current_instances)
                    except Exception as e:
                        logger.error(f"Scale-down callback error: {e}")
            
            # Record scaling action
            self.last_scaling_action = current_time
            self.scaling_history.append({
                "timestamp": current_time,
                "action": "scale_up" if decision > 0 else "scale_down",
                "old_instances": old_instances if 'old_instances' in locals() else self.current_instances,
                "new_instances": self.current_instances
            })

class ResourceManager:
    """Manages resource allocation and scaling"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.auto_scaler = AutoScaler(self.metrics_collector)
        self.resource_pools = {}
        self._setup_default_scaling_rules()
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules"""
        # CPU-based scaling
        cpu_rule = ScalingRule(
            metric_name="cpu_usage",
            threshold_up=80.0,
            threshold_down=30.0,
            cooldown_period=120,
            min_instances=1,
            max_instances=10,
            scale_up_step=2,
            scale_down_step=1
        )
        self.auto_scaler.add_scaling_rule(cpu_rule)
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            metric_name="memory_usage",
            threshold_up=85.0,
            threshold_down=40.0,
            cooldown_period=180,
            min_instances=1,
            max_instances=8,
            scale_up_step=1,
            scale_down_step=1
        )
        self.auto_scaler.add_scaling_rule(memory_rule)
        
        # Queue length-based scaling
        queue_rule = ScalingRule(
            metric_name="queue_length",
            threshold_up=50.0,
            threshold_down=5.0,
            cooldown_period=60,
            min_instances=1,
            max_instances=15,
            scale_up_step=3,
            scale_down_step=2
        )
        self.auto_scaler.add_scaling_rule(queue_rule)
    
    def start_auto_scaling(self):
        """Start auto-scaling"""
        self.auto_scaler.start_monitoring()
    
    def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.auto_scaler.stop_monitoring()
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        current_metrics = self.metrics_collector.current_metrics
        
        return {
            "current_instances": self.auto_scaler.current_instances,
            "metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "active_tasks": current_metrics.active_tasks,
                "queue_length": current_metrics.queue_length,
                "response_time": current_metrics.response_time,
                "error_rate": current_metrics.error_rate
            },
            "scaling_history": list(self.auto_scaler.scaling_history)[-10:],  # Last 10 actions
            "last_scaling_action": self.auto_scaler.last_scaling_action
        }

# Global resource manager
resource_manager = ResourceManager()
