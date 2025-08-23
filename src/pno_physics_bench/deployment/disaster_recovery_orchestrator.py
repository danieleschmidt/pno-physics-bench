# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Disaster Recovery Orchestrator for PNO Physics Bench

Manages disaster recovery and regional failover with:
- Automated failover detection and response
- Cross-region backup and recovery
- Business continuity planning
- RTO/RPO compliance
- Automated health monitoring
- Incident response automation
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .global_deployment_orchestrator import GlobalDeploymentOrchestrator, DeploymentStatus
from .global_data_synchronizer import GlobalDataSynchronizer
from .global_cdn_manager import GlobalCDNManager
from ..monitoring.global_monitoring_dashboard import GlobalMonitoringDashboard, AlertSeverity
from ..compliance import ComplianceManager
from ..i18n import get_text


class DisasterType(str, Enum):
    """Types of disasters that can occur."""
    REGION_OUTAGE = "region_outage"
    DATA_CENTER_FAILURE = "data_center_failure"
    NETWORK_PARTITION = "network_partition"
    APPLICATION_FAILURE = "application_failure"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RecoveryStrategy(str, Enum):
    """Disaster recovery strategies."""
    AUTOMATED_FAILOVER = "automated_failover"
    MANUAL_FAILOVER = "manual_failover"
    DATA_RESTORATION = "data_restoration"
    SERVICE_RESTART = "service_restart"
    ROLLBACK = "rollback"
    ISOLATION = "isolation"


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DisasterScenario:
    """Disaster recovery scenario definition."""
    
    scenario_id: str
    disaster_type: DisasterType
    affected_regions: List[str]
    severity: IncidentSeverity
    detection_criteria: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    failover_regions: List[str]
    automated_response: bool
    escalation_required: bool


@dataclass
class DisasterEvent:
    """Active disaster event."""
    
    event_id: str
    scenario_id: str
    disaster_type: DisasterType
    affected_regions: List[str]
    severity: IncidentSeverity
    detected_at: datetime
    recovery_started_at: Optional[datetime]
    recovery_completed_at: Optional[datetime]
    status: str
    recovery_actions: List[Dict[str, Any]]
    estimated_rto: int
    estimated_rpo: int
    current_impact: Dict[str, Any]


@dataclass
class RecoveryAction:
    """Individual recovery action."""
    
    action_id: str
    event_id: str
    action_type: str
    description: str
    target_region: str
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[str]
    error_message: Optional[str]


class DisasterRecoveryOrchestrator:
    """Orchestrates disaster recovery and regional failover."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.deployment_orchestrator = GlobalDeploymentOrchestrator()
        self.data_synchronizer = GlobalDataSynchronizer()
        self.cdn_manager = GlobalCDNManager()
        self.monitoring_dashboard = GlobalMonitoringDashboard()
        self.compliance_manager = ComplianceManager()
        
        # Disaster scenarios
        self.disaster_scenarios = self._initialize_disaster_scenarios()
        
        # Active events
        self.active_events: Dict[str, DisasterEvent] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # Configuration
        self.config = {
            "detection_interval_seconds": 30,
            "health_check_timeout_seconds": 10,
            "failover_timeout_seconds": 300,  # 5 minutes
            "recovery_confirmation_timeout": 600,  # 10 minutes
            "max_concurrent_recoveries": 3
        }
        
        # Service health thresholds
        self.health_thresholds = {
            "availability": 95.0,
            "response_time_ms": 2000,
            "error_rate_percent": 5.0,
            "cpu_utilization": 90.0,
            "memory_utilization": 90.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for disaster recovery."""
        
        logger = logging.getLogger("pno_disaster_recovery")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            # Critical incidents file
            incident_handler = logging.FileHandler("pno_disaster_recovery.log")
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            incident_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(incident_handler)
        
        return logger
    
    def _initialize_disaster_scenarios(self) -> List[DisasterScenario]:
        """Initialize predefined disaster recovery scenarios."""
        
        return [
            # Complete regional outage
            DisasterScenario(
                scenario_id="REGION_OUTAGE_EU",
                disaster_type=DisasterType.REGION_OUTAGE,
                affected_regions=["eu-west-1"],
                severity=IncidentSeverity.CRITICAL,
                detection_criteria={
                    "availability_threshold": 50.0,
                    "response_time_threshold": 5000,
                    "consecutive_failures": 3
                },
                recovery_strategy=RecoveryStrategy.AUTOMATED_FAILOVER,
                rto_minutes=15,
                rpo_minutes=5,
                failover_regions=["eu-central-1", "us-east-1"],
                automated_response=True,
                escalation_required=True
            ),
            
            # Application failure
            DisasterScenario(
                scenario_id="APP_FAILURE_CRITICAL",
                disaster_type=DisasterType.APPLICATION_FAILURE,
                affected_regions=["all"],
                severity=IncidentSeverity.HIGH,
                detection_criteria={
                    "error_rate_threshold": 10.0,
                    "availability_threshold": 80.0,
                    "duration_minutes": 5
                },
                recovery_strategy=RecoveryStrategy.SERVICE_RESTART,
                rto_minutes=10,
                rpo_minutes=1,
                failover_regions=[],
                automated_response=True,
                escalation_required=False
            ),
            
            # Data corruption
            DisasterScenario(
                scenario_id="DATA_CORRUPTION",
                disaster_type=DisasterType.DATA_CORRUPTION,
                affected_regions=["any"],
                severity=IncidentSeverity.CRITICAL,
                detection_criteria={
                    "data_integrity_failures": 3,
                    "checksum_mismatches": 5
                },
                recovery_strategy=RecoveryStrategy.DATA_RESTORATION,
                rto_minutes=60,
                rpo_minutes=15,
                failover_regions=[],
                automated_response=False,
                escalation_required=True
            ),
            
            # Security breach
            DisasterScenario(
                scenario_id="SECURITY_BREACH",
                disaster_type=DisasterType.SECURITY_BREACH,
                affected_regions=["any"],
                severity=IncidentSeverity.CRITICAL,
                detection_criteria={
                    "unauthorized_access": 1,
                    "suspicious_activity": 1
                },
                recovery_strategy=RecoveryStrategy.ISOLATION,
                rto_minutes=5,
                rpo_minutes=0,
                failover_regions=[],
                automated_response=True,
                escalation_required=True
            ),
            
            # Compliance violation
            DisasterScenario(
                scenario_id="COMPLIANCE_VIOLATION",
                disaster_type=DisasterType.COMPLIANCE_VIOLATION,
                affected_regions=["any"],
                severity=IncidentSeverity.HIGH,
                detection_criteria={
                    "compliance_score_threshold": 85.0,
                    "audit_failures": 3
                },
                recovery_strategy=RecoveryStrategy.ISOLATION,
                rto_minutes=30,
                rpo_minutes=5,
                failover_regions=[],
                automated_response=False,
                escalation_required=True
            )
        ]
    
    async def start_disaster_recovery_service(self):
        """Start the disaster recovery monitoring and response service."""
        
        self.logger.info("Starting disaster recovery service")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._disaster_detection_loop()),
            asyncio.create_task(self._recovery_execution_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._incident_management_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Disaster recovery service error: {e}")
            raise
    
    async def _disaster_detection_loop(self):
        """Continuously monitor for disaster conditions."""
        
        while True:
            try:
                await self._detect_disasters()
                await asyncio.sleep(self.config["detection_interval_seconds"])
            except Exception as e:
                self.logger.error(f"Disaster detection error: {e}")
                await asyncio.sleep(10)  # Short retry delay for critical monitoring
    
    async def _detect_disasters(self):
        """Detect disaster conditions based on system metrics."""
        
        # Get current system status
        deployment_status = await self.deployment_orchestrator.get_global_status()
        dashboard_data = await self.monitoring_dashboard.get_dashboard_data()
        
        # Check each scenario
        for scenario in self.disaster_scenarios:
            if await self._evaluate_disaster_criteria(scenario, deployment_status, dashboard_data):
                # Check if already handling this type of event
                if not self._is_event_active(scenario):
                    await self._trigger_disaster_response(scenario, deployment_status, dashboard_data)
    
    async def _evaluate_disaster_criteria(
        self,
        scenario: DisasterScenario,
        deployment_status: Dict[str, Any],
        dashboard_data: Dict[str, Any]
    ) -> bool:
        """Evaluate if disaster criteria are met."""
        
        criteria = scenario.detection_criteria
        
        # Region outage detection
        if scenario.disaster_type == DisasterType.REGION_OUTAGE:
            for region in scenario.affected_regions:
                regional_status = deployment_status.get("regional_status", {}).get(region, {})
                health_score = regional_status.get("health_score", 100)
                
                if health_score < criteria.get("availability_threshold", 50):
                    return True
        
        # Application failure detection
        elif scenario.disaster_type == DisasterType.APPLICATION_FAILURE:
            global_overview = dashboard_data.get("global_overview", {})
            error_rate = global_overview.get("global_error_rate_percent", 0)
            availability = deployment_status.get("healthy_regions", 0) / max(deployment_status.get("total_regions", 1), 1) * 100
            
            if (error_rate > criteria.get("error_rate_threshold", 10) or 
                availability < criteria.get("availability_threshold", 80)):
                return True
        
        # Data corruption detection
        elif scenario.disaster_type == DisasterType.DATA_CORRUPTION:
            # This would check data integrity metrics from the data synchronizer
            sync_status = await self.data_synchronizer.get_synchronization_status()
            failed_jobs = sync_status.get("sync_jobs", {}).get("failed", 0)
            
            if failed_jobs > criteria.get("data_integrity_failures", 3):
                return True
        
        # Security breach detection
        elif scenario.disaster_type == DisasterType.SECURITY_BREACH:
            # This would integrate with security monitoring systems
            # For simulation, we'll use a placeholder check
            return False  # No active security breaches in simulation
        
        # Compliance violation detection
        elif scenario.disaster_type == DisasterType.COMPLIANCE_VIOLATION:
            compliance_data = dashboard_data.get("compliance_status", {})
            compliance_score = compliance_data.get("global_compliance_score", 100)
            
            if compliance_score < criteria.get("compliance_score_threshold", 85):
                return True
        
        return False
    
    def _is_event_active(self, scenario: DisasterScenario) -> bool:
        """Check if a similar disaster event is already active."""
        
        for event in self.active_events.values():
            if (event.disaster_type == scenario.disaster_type and 
                event.status in ["detected", "recovery_in_progress"]):
                return True
        
        return False
    
    async def _trigger_disaster_response(
        self,
        scenario: DisasterScenario,
        deployment_status: Dict[str, Any],
        dashboard_data: Dict[str, Any]
    ):
        """Trigger disaster response for a scenario."""
        
        event_id = f"DR-{scenario.disaster_type.value}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        # Create disaster event
        disaster_event = DisasterEvent(
            event_id=event_id,
            scenario_id=scenario.scenario_id,
            disaster_type=scenario.disaster_type,
            affected_regions=scenario.affected_regions,
            severity=scenario.severity,
            detected_at=datetime.utcnow(),
            recovery_started_at=None,
            recovery_completed_at=None,
            status="detected",
            recovery_actions=[],
            estimated_rto=scenario.rto_minutes,
            estimated_rpo=scenario.rpo_minutes,
            current_impact=self._assess_current_impact(scenario, deployment_status, dashboard_data)
        )
        
        self.active_events[event_id] = disaster_event
        
        # Log critical incident
        self.logger.critical(f"DISASTER DETECTED: {scenario.disaster_type.value} - Event ID: {event_id}")
        self.logger.critical(f"Affected regions: {scenario.affected_regions}")
        self.logger.critical(f"Severity: {scenario.severity.value}")
        self.logger.critical(f"Estimated RTO: {scenario.rto_minutes} minutes")
        
        # Send immediate alerts
        await self._send_disaster_alert(disaster_event, scenario)
        
        # Start recovery if automated
        if scenario.automated_response:
            await self._initiate_recovery(disaster_event, scenario)
        else:
            self.logger.warning(f"Manual recovery required for event: {event_id}")
    
    def _assess_current_impact(
        self,
        scenario: DisasterScenario,
        deployment_status: Dict[str, Any],
        dashboard_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess current impact of the disaster."""
        
        impact = {
            "affected_users": 0,
            "affected_services": [],
            "data_at_risk": False,
            "compliance_impact": False,
            "business_impact": "low"
        }
        
        if scenario.disaster_type == DisasterType.REGION_OUTAGE:
            # Estimate users affected based on regional distribution
            total_users = dashboard_data.get("global_overview", {}).get("active_users", 0)
            region_user_ratio = 1.0 / max(deployment_status.get("total_regions", 1), 1)
            impact["affected_users"] = int(total_users * region_user_ratio * len(scenario.affected_regions))
            impact["business_impact"] = "high"
        
        elif scenario.disaster_type == DisasterType.APPLICATION_FAILURE:
            impact["affected_users"] = dashboard_data.get("global_overview", {}).get("active_users", 0)
            impact["affected_services"] = ["pno_physics_bench"]
            impact["business_impact"] = "critical"
        
        elif scenario.disaster_type == DisasterType.DATA_CORRUPTION:
            impact["data_at_risk"] = True
            impact["business_impact"] = "critical"
        
        elif scenario.disaster_type == DisasterType.COMPLIANCE_VIOLATION:
            impact["compliance_impact"] = True
            impact["business_impact"] = "high"
        
        return impact
    
    async def _send_disaster_alert(self, event: DisasterEvent, scenario: DisasterScenario):
        """Send disaster alert notifications."""
        
        alert = {
            "type": "DISASTER_ALERT",
            "event_id": event.event_id,
            "disaster_type": event.disaster_type.value,
            "severity": event.severity.value,
            "affected_regions": event.affected_regions,
            "estimated_rto": event.estimated_rto,
            "estimated_rpo": event.estimated_rpo,
            "current_impact": event.current_impact,
            "automated_response": scenario.automated_response,
            "timestamp": event.detected_at.isoformat()
        }
        
        # In production, this would integrate with:
        # - PagerDuty for critical incidents
        # - SMS/Phone for executives
        # - Slack for engineering teams
        # - Email for stakeholders
        
        self.logger.critical(f"DISASTER ALERT SENT: {json.dumps(alert, indent=2)}")
    
    async def _recovery_execution_loop(self):
        """Execute recovery actions for active events."""
        
        while True:
            try:
                await self._process_recovery_actions()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Recovery execution error: {e}")
                await asyncio.sleep(30)
    
    async def _process_recovery_actions(self):
        """Process pending recovery actions."""
        
        # Get events requiring recovery
        events_needing_recovery = [
            event for event in self.active_events.values()
            if event.status in ["detected", "recovery_in_progress"]
        ]
        
        # Limit concurrent recoveries
        active_recoveries = len([e for e in events_needing_recovery if e.status == "recovery_in_progress"])
        
        for event in events_needing_recovery:
            if active_recoveries >= self.config["max_concurrent_recoveries"]:
                break
            
            if event.status == "detected":
                scenario = next((s for s in self.disaster_scenarios if s.scenario_id == event.scenario_id), None)
                if scenario:
                    await self._initiate_recovery(event, scenario)
                    active_recoveries += 1
    
    async def _initiate_recovery(self, event: DisasterEvent, scenario: DisasterScenario):
        """Initiate recovery process for a disaster event."""
        
        event.status = "recovery_in_progress"
        event.recovery_started_at = datetime.utcnow()
        
        self.logger.info(f"Initiating recovery for event: {event.event_id}")
        
        try:
            if scenario.recovery_strategy == RecoveryStrategy.AUTOMATED_FAILOVER:
                await self._execute_automated_failover(event, scenario)
            
            elif scenario.recovery_strategy == RecoveryStrategy.SERVICE_RESTART:
                await self._execute_service_restart(event, scenario)
            
            elif scenario.recovery_strategy == RecoveryStrategy.DATA_RESTORATION:
                await self._execute_data_restoration(event, scenario)
            
            elif scenario.recovery_strategy == RecoveryStrategy.ISOLATION:
                await self._execute_isolation(event, scenario)
            
            elif scenario.recovery_strategy == RecoveryStrategy.ROLLBACK:
                await self._execute_rollback(event, scenario)
            
            else:
                raise ValueError(f"Unknown recovery strategy: {scenario.recovery_strategy}")
            
            # Mark recovery as completed
            event.status = "recovery_completed"
            event.recovery_completed_at = datetime.utcnow()
            
            # Calculate actual RTO
            actual_rto = (event.recovery_completed_at - event.detected_at).total_seconds() / 60
            
            self.logger.info(f"Recovery completed for event: {event.event_id}")
            self.logger.info(f"Actual RTO: {actual_rto:.1f} minutes (Target: {event.estimated_rto} minutes)")
            
            # Send recovery notification
            await self._send_recovery_notification(event, actual_rto)
            
        except Exception as e:
            event.status = "recovery_failed"
            self.logger.error(f"Recovery failed for event {event.event_id}: {e}")
            
            # Send failure notification
            await self._send_recovery_failure_notification(event, str(e))
    
    async def _execute_automated_failover(self, event: DisasterEvent, scenario: DisasterScenario):
        """Execute automated failover to backup regions."""
        
        self.logger.info(f"Executing automated failover for regions: {event.affected_regions}")
        
        # For each affected region, failover to backup regions
        for affected_region in event.affected_regions:
            # Select best failover region
            failover_region = await self._select_failover_region(affected_region, scenario.failover_regions)
            
            if failover_region:
                # Create recovery action
                action = await self._create_recovery_action(
                    event.event_id,
                    "automated_failover",
                    f"Failover from {affected_region} to {failover_region}",
                    failover_region
                )
                
                # Execute failover
                await self.deployment_orchestrator.failover_region(affected_region, failover_region)
                
                # Update DNS/CDN routing
                await self.cdn_manager.invalidate_cache([affected_region], ["/*"])
                
                # Mark action as completed
                action.status = "completed"
                action.completed_at = datetime.utcnow()
                action.result = f"Failover completed from {affected_region} to {failover_region}"
                
                event.recovery_actions.append(asdict(action))
                
                self.logger.info(f"Failover completed: {affected_region} -> {failover_region}")
            else:
                raise ValueError(f"No suitable failover region found for {affected_region}")
    
    async def _select_failover_region(self, affected_region: str, candidate_regions: List[str]) -> Optional[str]:
        """Select the best failover region."""
        
        # Get current deployment status
        deployment_status = await self.deployment_orchestrator.get_global_status()
        regional_status = deployment_status.get("regional_status", {})
        
        # Score candidate regions
        best_region = None
        best_score = 0
        
        for candidate in candidate_regions:
            if candidate == affected_region:
                continue
            
            region_status = regional_status.get(candidate, {})
            
            # Calculate score based on health, capacity, and compliance
            health_score = region_status.get("health_score", 0)
            replica_count = region_status.get("replica_count", 0)
            
            # Prefer regions with high health and available capacity
            score = health_score * 0.7 + min(replica_count * 10, 100) * 0.3
            
            if score > best_score:
                best_score = score
                best_region = candidate
        
        return best_region
    
    async def _execute_service_restart(self, event: DisasterEvent, scenario: DisasterScenario):
        """Execute service restart for application failures."""
        
        self.logger.info(f"Executing service restart for event: {event.event_id}")
        
        # Create recovery action
        action = await self._create_recovery_action(
            event.event_id,
            "service_restart",
            "Restart PNO Physics Bench services",
            "global"
        )
        
        # Simulate service restart (in production, this would restart actual services)
        await asyncio.sleep(30)  # Simulate restart time
        
        # Mark action as completed
        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.result = "Services restarted successfully"
        
        event.recovery_actions.append(asdict(action))
        
        self.logger.info("Service restart completed")
    
    async def _execute_data_restoration(self, event: DisasterEvent, scenario: DisasterScenario):
        """Execute data restoration from backups."""
        
        self.logger.info(f"Executing data restoration for event: {event.event_id}")
        
        # Create recovery action
        action = await self._create_recovery_action(
            event.event_id,
            "data_restoration",
            "Restore data from latest backups",
            "global"
        )
        
        # Simulate data restoration (in production, this would restore from actual backups)
        await asyncio.sleep(120)  # Simulate restore time
        
        # Mark action as completed
        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.result = "Data restored from backup"
        
        event.recovery_actions.append(asdict(action))
        
        self.logger.info("Data restoration completed")
    
    async def _execute_isolation(self, event: DisasterEvent, scenario: DisasterScenario):
        """Execute isolation for security breaches or compliance violations."""
        
        self.logger.info(f"Executing isolation for event: {event.event_id}")
        
        # Create recovery action
        action = await self._create_recovery_action(
            event.event_id,
            "isolation",
            "Isolate affected systems",
            "global"
        )
        
        # Simulate isolation (in production, this would isolate actual systems)
        await asyncio.sleep(10)  # Simulate isolation time
        
        # Mark action as completed
        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.result = "Systems isolated successfully"
        
        event.recovery_actions.append(asdict(action))
        
        self.logger.info("Isolation completed")
    
    async def _execute_rollback(self, event: DisasterEvent, scenario: DisasterScenario):
        """Execute rollback to previous version."""
        
        self.logger.info(f"Executing rollback for event: {event.event_id}")
        
        # Create recovery action
        action = await self._create_recovery_action(
            event.event_id,
            "rollback",
            "Rollback to previous stable version",
            "global"
        )
        
        # Simulate rollback (in production, this would rollback actual deployments)
        await asyncio.sleep(60)  # Simulate rollback time
        
        # Mark action as completed
        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.result = "Rollback completed successfully"
        
        event.recovery_actions.append(asdict(action))
        
        self.logger.info("Rollback completed")
    
    async def _create_recovery_action(
        self,
        event_id: str,
        action_type: str,
        description: str,
        target_region: str
    ) -> RecoveryAction:
        """Create a recovery action."""
        
        action_id = f"RA-{action_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        action = RecoveryAction(
            action_id=action_id,
            event_id=event_id,
            action_type=action_type,
            description=description,
            target_region=target_region,
            status="in_progress",
            started_at=datetime.utcnow(),
            completed_at=None,
            result=None,
            error_message=None
        )
        
        self.recovery_actions[action_id] = action
        
        return action
    
    async def _send_recovery_notification(self, event: DisasterEvent, actual_rto: float):
        """Send recovery completion notification."""
        
        notification = {
            "type": "RECOVERY_COMPLETED",
            "event_id": event.event_id,
            "disaster_type": event.disaster_type.value,
            "recovery_started_at": event.recovery_started_at.isoformat(),
            "recovery_completed_at": event.recovery_completed_at.isoformat(),
            "actual_rto_minutes": round(actual_rto, 1),
            "target_rto_minutes": event.estimated_rto,
            "rto_met": actual_rto <= event.estimated_rto,
            "recovery_actions": len(event.recovery_actions)
        }
        
        self.logger.info(f"RECOVERY NOTIFICATION: {json.dumps(notification, indent=2)}")
    
    async def _send_recovery_failure_notification(self, event: DisasterEvent, error: str):
        """Send recovery failure notification."""
        
        notification = {
            "type": "RECOVERY_FAILED",
            "event_id": event.event_id,
            "disaster_type": event.disaster_type.value,
            "error": error,
            "escalation_required": True,
            "manual_intervention_needed": True
        }
        
        self.logger.critical(f"RECOVERY FAILURE NOTIFICATION: {json.dumps(notification, indent=2)}")
    
    async def _health_monitoring_loop(self):
        """Monitor system health for recovery validation."""
        
        while True:
            try:
                await self._validate_recovery_health()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _validate_recovery_health(self):
        """Validate that recovered systems are healthy."""
        
        # Get events that recently completed recovery
        recently_recovered = [
            event for event in self.active_events.values()
            if (event.status == "recovery_completed" and
                event.recovery_completed_at and
                (datetime.utcnow() - event.recovery_completed_at).total_seconds() < 1800)  # Last 30 minutes
        ]
        
        for event in recently_recovered:
            # Check if the recovered systems are stable
            deployment_status = await self.deployment_orchestrator.get_global_status()
            
            # If system is healthy, mark event as resolved
            if deployment_status.get("global_status") == "healthy":
                event.status = "resolved"
                self.logger.info(f"Event {event.event_id} marked as resolved - system stable")
    
    async def _incident_management_loop(self):
        """Manage incident lifecycle and cleanup."""
        
        while True:
            try:
                await self._cleanup_old_events()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Incident management error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_events(self):
        """Clean up old resolved events."""
        
        cutoff_time = datetime.utcnow() - timedelta(days=30)  # Keep events for 30 days
        
        events_to_remove = []
        for event_id, event in self.active_events.items():
            if (event.status in ["resolved", "recovery_failed"] and
                event.detected_at < cutoff_time):
                events_to_remove.append(event_id)
        
        for event_id in events_to_remove:
            del self.active_events[event_id]
            self.logger.debug(f"Cleaned up old event: {event_id}")
    
    async def get_disaster_recovery_status(self) -> Dict[str, Any]:
        """Get current disaster recovery status."""
        
        # Event statistics
        total_events = len(self.active_events)
        active_events = len([e for e in self.active_events.values() if e.status in ["detected", "recovery_in_progress"]])
        resolved_events = len([e for e in self.active_events.values() if e.status == "resolved"])
        failed_recoveries = len([e for e in self.active_events.values() if e.status == "recovery_failed"])
        
        # Recent events
        recent_events = sorted(
            self.active_events.values(),
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        # RTO/RPO compliance
        completed_events = [e for e in self.active_events.values() if e.status in ["resolved", "recovery_completed"]]
        rto_compliance = []
        
        for event in completed_events:
            if event.recovery_completed_at:
                actual_rto = (event.recovery_completed_at - event.detected_at).total_seconds() / 60
                rto_met = actual_rto <= event.estimated_rto
                rto_compliance.append(rto_met)
        
        rto_compliance_rate = sum(rto_compliance) / len(rto_compliance) * 100 if rto_compliance else 100
        
        return {
            "last_updated": datetime.utcnow().isoformat(),
            "service_status": "active",
            "event_statistics": {
                "total_events": total_events,
                "active_events": active_events,
                "resolved_events": resolved_events,
                "failed_recoveries": failed_recoveries
            },
            "rto_compliance_rate": round(rto_compliance_rate, 1),
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "disaster_type": event.disaster_type.value,
                    "severity": event.severity.value,
                    "status": event.status,
                    "detected_at": event.detected_at.isoformat(),
                    "affected_regions": event.affected_regions
                }
                for event in recent_events
            ],
            "monitoring_active": True,
            "automated_response_enabled": True
        }


# Convenience functions
async def start_disaster_recovery():
    """Start the disaster recovery service."""
    
    orchestrator = DisasterRecoveryOrchestrator()
    await orchestrator.start_disaster_recovery_service()


async def get_dr_status() -> Dict[str, Any]:
    """Get disaster recovery status."""
    
    orchestrator = DisasterRecoveryOrchestrator()
    return await orchestrator.get_disaster_recovery_status()


__all__ = [
    "DisasterRecoveryOrchestrator",
    "DisasterType",
    "RecoveryStrategy",
    "IncidentSeverity",
    "DisasterScenario",
    "DisasterEvent",
    "RecoveryAction",
    "start_disaster_recovery",
    "get_dr_status"
]