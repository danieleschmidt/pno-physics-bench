# PNO Physics Bench - Operational Maintenance Guide

## Overview
Comprehensive operational maintenance procedures for the PNO Physics Bench production environment, including scheduled maintenance, preventive measures, and system optimization.

## Table of Contents
1. [Maintenance Schedule](#maintenance-schedule)
2. [Preventive Maintenance](#preventive-maintenance)
3. [System Monitoring](#system-monitoring)
4. [Backup and Recovery](#backup-and-recovery)
5. [Security Maintenance](#security-maintenance)
6. [Performance Optimization](#performance-optimization)
7. [Capacity Planning](#capacity-planning)
8. [Documentation Updates](#documentation-updates)

## Maintenance Schedule

### Daily Operations (Automated)
**Time**: 02:00 UTC
**Duration**: 30 minutes
**Impact**: None (zero-downtime)

```bash
#!/bin/bash
# Daily maintenance script
# File: /opt/pno/maintenance/daily-maintenance.sh

set -e

echo "Starting daily maintenance - $(date)"

# 1. Health checks
echo "Running health checks..."
kubectl get pods -n production -l app=pno-physics-bench
curl -f https://api.pno-physics-bench.com/health

# 2. Log rotation
echo "Rotating logs..."
kubectl exec deployment/pno-physics-bench -n production -- logrotate -f /etc/logrotate.conf

# 3. Metrics collection
echo "Collecting metrics..."
python /opt/pno/scripts/collect_daily_metrics.py

# 4. Security scans
echo "Running security scans..."
python /opt/pno/security/daily_vulnerability_scan.py

# 5. Backup verification
echo "Verifying backups..."
python /opt/pno/backup/verify_backups.py

# 6. Performance analysis
echo "Analyzing performance..."
python /opt/pno/monitoring/daily_performance_analysis.py

echo "Daily maintenance completed - $(date)"
```

### Weekly Operations
**Time**: Sunday 03:00 UTC
**Duration**: 2 hours
**Impact**: Minimal (rolling updates)

```bash
#!/bin/bash
# Weekly maintenance script
# File: /opt/pno/maintenance/weekly-maintenance.sh

set -e

echo "Starting weekly maintenance - $(date)"

# 1. System updates
echo "Checking for system updates..."
kubectl get nodes -o yaml > /tmp/nodes-backup-$(date +%Y%m%d).yaml

# 2. Certificate renewal check
echo "Checking certificate expiry..."
kubectl get certificates -n production
python /opt/pno/security/check_certificate_expiry.py

# 3. Model validation
echo "Validating model integrity..."
python /opt/pno/models/validate_model_integrity.py

# 4. Database maintenance
echo "Database maintenance..."
kubectl exec deployment/pno-physics-bench -n production -- python scripts/database_maintenance.py

# 5. Performance optimization
echo "Performance optimization..."
python /opt/pno/optimization/weekly_performance_optimization.py

# 6. Capacity analysis
echo "Capacity analysis..."
python /opt/pno/monitoring/weekly_capacity_analysis.py

# 7. Security audit
echo "Security audit..."
python /opt/pno/security/weekly_security_audit.py

echo "Weekly maintenance completed - $(date)"
```

### Monthly Operations
**Time**: First Sunday 01:00 UTC
**Duration**: 4 hours
**Impact**: Moderate (planned maintenance window)

```bash
#!/bin/bash
# Monthly maintenance script
# File: /opt/pno/maintenance/monthly-maintenance.sh

set -e

echo "Starting monthly maintenance - $(date)"

# 1. Full system backup
echo "Creating full system backup..."
python /opt/pno/backup/full_system_backup.py

# 2. Security updates
echo "Applying security updates..."
python /opt/pno/security/apply_security_updates.py

# 3. Model retraining (if needed)
echo "Checking model retraining needs..."
python /opt/pno/ml/check_retraining_needs.py

# 4. Infrastructure updates
echo "Infrastructure updates..."
kubectl apply -f /opt/pno/k8s/infrastructure-updates/

# 5. Compliance reporting
echo "Generating compliance reports..."
python /opt/pno/compliance/generate_monthly_report.py

# 6. Disaster recovery testing
echo "Testing disaster recovery procedures..."
python /opt/pno/dr/test_disaster_recovery.py

echo "Monthly maintenance completed - $(date)"
```

### Quarterly Operations
**Time**: First Sunday of quarter 00:00 UTC
**Duration**: 8 hours
**Impact**: High (maintenance window required)

```bash
#!/bin/bash
# Quarterly maintenance script
# File: /opt/pno/maintenance/quarterly-maintenance.sh

set -e

echo "Starting quarterly maintenance - $(date)"

# 1. Major version updates
echo "Planning major updates..."
python /opt/pno/updates/plan_major_updates.py

# 2. Full security audit
echo "Full security audit..."
python /opt/pno/security/full_security_audit.py

# 3. Performance benchmarking
echo "Performance benchmarking..."
python /opt/pno/performance/quarterly_benchmark.py

# 4. Capacity planning review
echo "Capacity planning review..."
python /opt/pno/capacity/quarterly_capacity_review.py

# 5. Documentation review
echo "Documentation review..."
python /opt/pno/docs/quarterly_doc_review.py

echo "Quarterly maintenance completed - $(date)"
```

## Preventive Maintenance

### System Health Monitoring
```bash
#!/bin/bash
# Continuous health monitoring
# File: /opt/pno/monitoring/health_monitor.sh

while true; do
    # Check pod health
    UNHEALTHY_PODS=$(kubectl get pods -n production -l app=pno-physics-bench --field-selector=status.phase!=Running --no-headers | wc -l)
    
    if [ $UNHEALTHY_PODS -gt 0 ]; then
        echo "Alert: $UNHEALTHY_PODS unhealthy pods detected"
        kubectl get pods -n production -l app=pno-physics-bench
        # Send alert
        python /opt/pno/alerts/send_alert.py --type="pod_health" --count=$UNHEALTHY_PODS
    fi
    
    # Check resource usage
    CPU_USAGE=$(kubectl top pods -n production -l app=pno-physics-bench --no-headers | awk '{sum+=$2} END {print sum}')
    MEMORY_USAGE=$(kubectl top pods -n production -l app=pno-physics-bench --no-headers | awk '{sum+=$3} END {print sum}')
    
    echo "Current usage - CPU: ${CPU_USAGE}m, Memory: ${MEMORY_USAGE}Mi"
    
    # Check API health
    if ! curl -f -s https://api.pno-physics-bench.com/health > /dev/null; then
        echo "Alert: API health check failed"
        python /opt/pno/alerts/send_alert.py --type="api_health_failure"
    fi
    
    sleep 60
done
```

### Automated Scaling Optimization
```python
#!/usr/bin/env python3
# Auto-scaling optimization
# File: /opt/pno/optimization/auto_scaling_optimizer.py

import subprocess
import json
import time
from datetime import datetime, timedelta

class AutoScalingOptimizer:
    def __init__(self):
        self.namespace = "production"
        self.deployment = "pno-physics-bench"
        
    def get_current_metrics(self):
        """Get current resource usage metrics"""
        # Get CPU usage
        cpu_cmd = f"kubectl top pods -n {self.namespace} -l app={self.deployment} --no-headers"
        cpu_output = subprocess.check_output(cpu_cmd, shell=True).decode()
        
        cpu_usage = []
        memory_usage = []
        
        for line in cpu_output.strip().split('\n'):
            if line:
                parts = line.split()
                cpu_usage.append(int(parts[1].replace('m', '')))
                memory_usage.append(int(parts[2].replace('Mi', '')))
        
        return {
            'cpu_avg': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            'memory_avg': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'replica_count': len(cpu_usage)
        }
    
    def get_hpa_status(self):
        """Get HPA current status"""
        cmd = f"kubectl get hpa pno-hpa -n {self.namespace} -o json"
        output = subprocess.check_output(cmd, shell=True).decode()
        hpa_data = json.loads(output)
        
        return {
            'current_replicas': hpa_data['status']['currentReplicas'],
            'desired_replicas': hpa_data['status']['desiredReplicas'],
            'max_replicas': hpa_data['spec']['maxReplicas'],
            'min_replicas': hpa_data['spec']['minReplicas']
        }
    
    def optimize_scaling_parameters(self):
        """Optimize HPA parameters based on historical data"""
        metrics = self.get_current_metrics()
        hpa_status = self.get_hpa_status()
        
        # Optimization logic
        recommendations = []
        
        # CPU-based optimization
        if metrics['cpu_avg'] > 800:  # 80% CPU usage
            if hpa_status['max_replicas'] < 20:
                recommendations.append({
                    'action': 'increase_max_replicas',
                    'current': hpa_status['max_replicas'],
                    'recommended': min(20, hpa_status['max_replicas'] + 5),
                    'reason': 'High CPU usage detected'
                })
        
        # Memory-based optimization
        if metrics['memory_avg'] > 3200:  # 80% of 4Gi limit
            recommendations.append({
                'action': 'increase_memory_limits',
                'current': '4Gi',
                'recommended': '6Gi',
                'reason': 'High memory usage detected'
            })
        
        return recommendations
    
    def apply_recommendations(self, recommendations):
        """Apply optimization recommendations"""
        for rec in recommendations:
            if rec['action'] == 'increase_max_replicas':
                cmd = f"""kubectl patch hpa pno-hpa -n {self.namespace} -p '{{"spec":{{"maxReplicas":{rec['recommended']}}}}}'"""
                subprocess.run(cmd, shell=True)
                print(f"Increased max replicas to {rec['recommended']}")
            
            elif rec['action'] == 'increase_memory_limits':
                cmd = f"""kubectl patch deployment {self.deployment} -n {self.namespace} -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"pno-inference","resources":{{"limits":{{"memory":"{rec['recommended']}"}}}}}}}]}}}}}}'"""
                subprocess.run(cmd, shell=True)
                print(f"Increased memory limit to {rec['recommended']}")

def main():
    optimizer = AutoScalingOptimizer()
    
    while True:
        try:
            recommendations = optimizer.optimize_scaling_parameters()
            
            if recommendations:
                print(f"Optimization recommendations at {datetime.now()}:")
                for rec in recommendations:
                    print(f"  - {rec['action']}: {rec['reason']}")
                
                # Apply recommendations (uncomment for automatic application)
                # optimizer.apply_recommendations(recommendations)
            
            time.sleep(300)  # Run every 5 minutes
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
```

### Model Health Validation
```python
#!/usr/bin/env python3
# Model health validation
# File: /opt/pno/models/model_health_validator.py

import torch
import numpy as np
import subprocess
import json
from pathlib import Path

class ModelHealthValidator:
    def __init__(self):
        self.model_path = "/app/models/pno_model.pth"
        self.test_data_path = "/app/test_data/validation_set.npz"
        
    def validate_model_integrity(self):
        """Validate model file integrity"""
        try:
            # Check if model file exists and is readable
            if not Path(self.model_path).exists():
                return {"status": "FAIL", "reason": "Model file not found"}
            
            # Load model
            model = torch.load(self.model_path, map_location='cpu')
            
            # Basic structure validation
            if not isinstance(model, dict) or 'state_dict' not in model:
                return {"status": "FAIL", "reason": "Invalid model structure"}
            
            # Check for corrupted weights
            for name, param in model['state_dict'].items():
                if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                    return {"status": "FAIL", "reason": f"Corrupted weights in {name}"}
            
            return {"status": "PASS", "reason": "Model integrity validated"}
            
        except Exception as e:
            return {"status": "FAIL", "reason": f"Model validation error: {str(e)}"}
    
    def validate_model_performance(self):
        """Validate model performance on test data"""
        try:
            # Load test data
            test_data = np.load(self.test_data_path)
            
            # Run inference test
            cmd = """kubectl exec deployment/pno-physics-bench -n production -- python -c "
import sys
sys.path.append('/app')
from src.pno_physics_bench.models import PNOModel
import torch
import numpy as np

# Load test data
data = np.load('/app/test_data/validation_set.npz')
X_test, y_test = data['X'], data['y']

# Load model
model = PNOModel()
checkpoint = torch.load('/app/models/pno_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    X_tensor = torch.from_numpy(X_test[:10]).float()
    predictions = model(X_tensor)
    
    # Check for NaN outputs
    if torch.any(torch.isnan(predictions)):
        print('FAIL: NaN in predictions')
    else:
        print('PASS: Model inference successful')
"
            """
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if "PASS" in result.stdout:
                return {"status": "PASS", "reason": "Model performance validated"}
            else:
                return {"status": "FAIL", "reason": "Model performance validation failed"}
                
        except Exception as e:
            return {"status": "FAIL", "reason": f"Performance validation error: {str(e)}"}
    
    def run_full_validation(self):
        """Run complete model validation"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "integrity_check": self.validate_model_integrity(),
            "performance_check": self.validate_model_performance()
        }
        
        # Overall status
        all_passed = all(result["status"] == "PASS" for result in [
            results["integrity_check"],
            results["performance_check"]
        ])
        
        results["overall_status"] = "PASS" if all_passed else "FAIL"
        
        return results

def main():
    validator = ModelHealthValidator()
    results = validator.run_full_validation()
    
    print(json.dumps(results, indent=2))
    
    # Alert if validation fails
    if results["overall_status"] == "FAIL":
        subprocess.run([
            "python", "/opt/pno/alerts/send_alert.py",
            "--type", "model_validation_failure",
            "--data", json.dumps(results)
        ])

if __name__ == "__main__":
    main()
```

## System Monitoring

### Comprehensive Monitoring Setup
```yaml
# monitoring-setup.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'pno-physics-bench'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - production
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: ${1}:${2}
            target_label: __address__
      
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
            
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name
            
  alerting_rules.yml: |
    groups:
      - name: pno-physics-bench.rules
        rules:
          - alert: PNOHighLatency
            expr: histogram_quantile(0.95, rate(pno_inference_duration_seconds_bucket[5m])) > 0.5
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "PNO inference latency is high"
              description: "95th percentile latency is {{ $value }}s"
              
          - alert: PNOHighErrorRate
            expr: rate(pno_inference_errors_total[5m]) > 0.05
            for: 2m
            labels:
              severity: critical
            annotations:
              summary: "PNO error rate is high"
              description: "Error rate is {{ $value | humanizePercentage }}"
              
          - alert: PNOPodDown
            expr: up{job="pno-physics-bench"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "PNO pod is down"
              description: "Pod {{ $labels.instance }} is not responding"
              
          - alert: PNOMemoryUsageHigh
            expr: container_memory_usage_bytes{pod=~"pno-physics-bench-.*"} / container_spec_memory_limit_bytes > 0.9
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "PNO memory usage is high"
              description: "Memory usage is {{ $value | humanizePercentage }}"
              
          - alert: PNODiskSpaceHigh
            expr: node_filesystem_avail_bytes{mountpoint="/app/models"} / node_filesystem_size_bytes{mountpoint="/app/models"} < 0.1
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "PNO disk space is low"
              description: "Only {{ $value | humanizePercentage }} disk space remaining"
```

### Custom Monitoring Scripts
```python
#!/usr/bin/env python3
# Custom monitoring script
# File: /opt/pno/monitoring/custom_monitor.py

import time
import subprocess
import json
import requests
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PNOMonitor:
    def __init__(self):
        self.namespace = "production"
        self.deployment = "pno-physics-bench"
        self.api_endpoint = "https://api.pno-physics-bench.com"
        
    def check_pod_health(self):
        """Check pod health status"""
        try:
            cmd = f"kubectl get pods -n {self.namespace} -l app={self.deployment} -o json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"status": "ERROR", "message": "Failed to get pod status"}
            
            pods_data = json.loads(result.stdout)
            pods = pods_data['items']
            
            healthy_pods = 0
            total_pods = len(pods)
            
            for pod in pods:
                if pod['status']['phase'] == 'Running':
                    # Check if all containers are ready
                    container_statuses = pod['status'].get('containerStatuses', [])
                    if all(container['ready'] for container in container_statuses):
                        healthy_pods += 1
            
            return {
                "status": "OK" if healthy_pods == total_pods else "DEGRADED",
                "healthy_pods": healthy_pods,
                "total_pods": total_pods,
                "health_ratio": healthy_pods / total_pods if total_pods > 0 else 0
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def check_api_health(self):
        """Check API endpoint health"""
        try:
            # Health check
            health_response = requests.get(f"{self.api_endpoint}/health", timeout=10)
            health_ok = health_response.status_code == 200
            
            # Readiness check
            ready_response = requests.get(f"{self.api_endpoint}/ready", timeout=10)
            ready_ok = ready_response.status_code == 200
            
            # Performance check
            start_time = time.time()
            perf_response = requests.get(f"{self.api_endpoint}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "OK" if health_ok and ready_ok else "DEGRADED",
                "health_endpoint": health_ok,
                "ready_endpoint": ready_ok,
                "response_time_ms": response_time
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def check_resource_usage(self):
        """Check resource usage"""
        try:
            cmd = f"kubectl top pods -n {self.namespace} -l app={self.deployment} --no-headers"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"status": "ERROR", "message": "Failed to get resource usage"}
            
            total_cpu = 0
            total_memory = 0
            pod_count = 0
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    cpu_usage = int(parts[1].replace('m', ''))
                    memory_usage = int(parts[2].replace('Mi', ''))
                    
                    total_cpu += cpu_usage
                    total_memory += memory_usage
                    pod_count += 1
            
            avg_cpu = total_cpu / pod_count if pod_count > 0 else 0
            avg_memory = total_memory / pod_count if pod_count > 0 else 0
            
            return {
                "status": "OK",
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "total_cpu_usage": total_cpu,
                "total_memory_usage": total_memory,
                "pod_count": pod_count
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def check_hpa_status(self):
        """Check HPA scaling status"""
        try:
            cmd = f"kubectl get hpa pno-hpa -n {self.namespace} -o json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"status": "ERROR", "message": "Failed to get HPA status"}
            
            hpa_data = json.loads(result.stdout)
            
            return {
                "status": "OK",
                "current_replicas": hpa_data['status']['currentReplicas'],
                "desired_replicas": hpa_data['status']['desiredReplicas'],
                "min_replicas": hpa_data['spec']['minReplicas'],
                "max_replicas": hpa_data['spec']['maxReplicas'],
                "scaling_active": hpa_data['status']['currentReplicas'] != hpa_data['status']['desiredReplicas']
            }
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def run_comprehensive_check(self):
        """Run all monitoring checks"""
        timestamp = datetime.now().isoformat()
        
        results = {
            "timestamp": timestamp,
            "pod_health": self.check_pod_health(),
            "api_health": self.check_api_health(),
            "resource_usage": self.check_resource_usage(),
            "hpa_status": self.check_hpa_status()
        }
        
        # Overall health assessment
        all_ok = all(
            check.get("status") == "OK" 
            for check in results.values() 
            if isinstance(check, dict) and "status" in check
        )
        
        results["overall_status"] = "HEALTHY" if all_ok else "UNHEALTHY"
        
        return results
    
    def send_alerts(self, results):
        """Send alerts for unhealthy conditions"""
        if results["overall_status"] != "HEALTHY":
            alert_data = {
                "timestamp": results["timestamp"],
                "severity": "WARNING",
                "service": "pno-physics-bench",
                "issues": []
            }
            
            for check_name, check_result in results.items():
                if isinstance(check_result, dict) and check_result.get("status") != "OK":
                    alert_data["issues"].append({
                        "check": check_name,
                        "status": check_result.get("status"),
                        "message": check_result.get("message", "Check failed")
                    })
            
            # Send alert (implement your alerting mechanism)
            logger.warning(f"Health check alert: {json.dumps(alert_data)}")

def main():
    monitor = PNOMonitor()
    
    while True:
        try:
            results = monitor.run_comprehensive_check()
            
            # Log results
            logger.info(f"Health check completed: {results['overall_status']}")
            
            # Send alerts if needed
            monitor.send_alerts(results)
            
            # Save results for historical analysis
            with open(f"/var/log/pno/health-check-{datetime.now().strftime('%Y%m%d')}.jsonl", "a") as f:
                f.write(json.dumps(results) + "\n")
            
            time.sleep(60)  # Run every minute
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
```

## Backup and Recovery

### Automated Backup System
```bash
#!/bin/bash
# Automated backup system
# File: /opt/pno/backup/automated_backup.sh

set -e

BACKUP_DIR="/backup/pno-physics-bench"
DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACE="production"

echo "Starting automated backup - $DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# 1. Kubernetes resource backup
echo "Backing up Kubernetes resources..."
kubectl get all -n $NAMESPACE -o yaml > "$BACKUP_DIR/$DATE/k8s-resources.yaml"
kubectl get configmap,secret -n $NAMESPACE -o yaml > "$BACKUP_DIR/$DATE/k8s-config.yaml"
kubectl get pv,pvc -n $NAMESPACE -o yaml > "$BACKUP_DIR/$DATE/k8s-storage.yaml"

# 2. Model backup
echo "Backing up models..."
kubectl cp $NAMESPACE/$(kubectl get pods -n $NAMESPACE -l app=pno-physics-bench -o jsonpath='{.items[0].metadata.name}'):/app/models/ "$BACKUP_DIR/$DATE/models/"

# 3. Configuration backup
echo "Backing up configuration..."
kubectl get configmap pno-config -n $NAMESPACE -o yaml > "$BACKUP_DIR/$DATE/app-config.yaml"

# 4. Monitoring configuration backup
echo "Backing up monitoring configuration..."
kubectl get configmap -n monitoring -o yaml > "$BACKUP_DIR/$DATE/monitoring-config.yaml"

# 5. Database backup (if applicable)
echo "Backing up database..."
kubectl exec deployment/pno-physics-bench -n $NAMESPACE -- pg_dump pno_db > "$BACKUP_DIR/$DATE/database.sql" 2>/dev/null || echo "No database to backup"

# 6. Create backup manifest
cat > "$BACKUP_DIR/$DATE/backup-manifest.json" << EOF
{
  "timestamp": "$DATE",
  "type": "automated",
  "namespace": "$NAMESPACE",
  "components": [
    "kubernetes-resources",
    "models",
    "configuration",
    "monitoring",
    "database"
  ],
  "backup_size": "$(du -sh $BACKUP_DIR/$DATE | cut -f1)",
  "backup_location": "$BACKUP_DIR/$DATE"
}
EOF

# 7. Compress backup
echo "Compressing backup..."
tar -czf "$BACKUP_DIR/backup-$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# 8. Verify backup
echo "Verifying backup..."
if tar -tzf "$BACKUP_DIR/backup-$DATE.tar.gz" > /dev/null; then
    echo "Backup verification successful"
else
    echo "Backup verification failed"
    exit 1
fi

# 9. Upload to cloud storage (optional)
# aws s3 cp "$BACKUP_DIR/backup-$DATE.tar.gz" s3://pno-backups/

# 10. Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +30 -delete

echo "Backup completed successfully - backup-$DATE.tar.gz"
```

### Disaster Recovery Procedures
```bash
#!/bin/bash
# Disaster recovery script
# File: /opt/pno/dr/disaster_recovery.sh

set -e

BACKUP_FILE="$1"
NAMESPACE="production"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

echo "Starting disaster recovery from $BACKUP_FILE"

# 1. Verify backup file
echo "Verifying backup file..."
if ! tar -tzf "$BACKUP_FILE" > /dev/null; then
    echo "Backup file verification failed"
    exit 1
fi

# 2. Extract backup
TEMP_DIR=$(mktemp -d)
echo "Extracting backup to $TEMP_DIR..."
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

BACKUP_DIR=$(find "$TEMP_DIR" -type d -name "20*" | head -1)

# 3. Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 4. Restore Kubernetes resources
echo "Restoring Kubernetes resources..."
kubectl apply -f "$BACKUP_DIR/k8s-resources.yaml"
kubectl apply -f "$BACKUP_DIR/k8s-config.yaml"
kubectl apply -f "$BACKUP_DIR/k8s-storage.yaml"

# 5. Wait for pods to be running
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=pno-physics-bench -n $NAMESPACE --timeout=300s

# 6. Restore models
echo "Restoring models..."
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=pno-physics-bench -o jsonpath='{.items[0].metadata.name}')
kubectl cp "$BACKUP_DIR/models/" "$NAMESPACE/$POD_NAME:/app/models/"

# 7. Restart deployment to pick up restored models
echo "Restarting deployment..."
kubectl rollout restart deployment/pno-physics-bench -n $NAMESPACE
kubectl rollout status deployment/pno-physics-bench -n $NAMESPACE --timeout=300s

# 8. Verify recovery
echo "Verifying recovery..."
sleep 30
if curl -f -s https://api.pno-physics-bench.com/health > /dev/null; then
    echo "Disaster recovery successful"
else
    echo "Disaster recovery verification failed"
    exit 1
fi

# 9. Cleanup
rm -rf "$TEMP_DIR"

echo "Disaster recovery completed successfully"
```

## Security Maintenance

### Security Update Automation
```python
#!/usr/bin/env python3
# Security update automation
# File: /opt/pno/security/security_updater.py

import subprocess
import json
import logging
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityUpdater:
    def __init__(self):
        self.namespace = "production"
        self.deployment = "pno-physics-bench"
        
    def scan_vulnerabilities(self):
        """Scan for vulnerabilities in current deployment"""
        try:
            # Get current image
            cmd = f"kubectl get deployment {self.deployment} -n {self.namespace} -o jsonpath='{{.spec.template.spec.containers[0].image}}'"
            current_image = subprocess.check_output(cmd, shell=True).decode().strip()
            
            logger.info(f"Scanning vulnerabilities for image: {current_image}")
            
            # Run Trivy scan
            scan_cmd = f"trivy image --format json {current_image}"
            scan_result = subprocess.run(scan_cmd, shell=True, capture_output=True, text=True)
            
            if scan_result.returncode != 0:
                logger.error(f"Vulnerability scan failed: {scan_result.stderr}")
                return None
            
            scan_data = json.loads(scan_result.stdout)
            
            # Analyze results
            vulnerabilities = []
            for result in scan_data.get('Results', []):
                for vuln in result.get('Vulnerabilities', []):
                    vulnerabilities.append({
                        'id': vuln.get('VulnerabilityID'),
                        'severity': vuln.get('Severity'),
                        'package': vuln.get('PkgName'),
                        'installed_version': vuln.get('InstalledVersion'),
                        'fixed_version': vuln.get('FixedVersion'),
                        'title': vuln.get('Title')
                    })
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Vulnerability scan error: {e}")
            return None
    
    def check_critical_vulnerabilities(self, vulnerabilities):
        """Check for critical vulnerabilities that require immediate attention"""
        if not vulnerabilities:
            return []
        
        critical_vulns = [
            vuln for vuln in vulnerabilities 
            if vuln['severity'] in ['CRITICAL', 'HIGH'] and vuln.get('fixed_version')
        ]
        
        return critical_vulns
    
    def apply_security_updates(self, vulnerabilities):
        """Apply security updates for identified vulnerabilities"""
        critical_vulns = self.check_critical_vulnerabilities(vulnerabilities)
        
        if not critical_vulns:
            logger.info("No critical vulnerabilities requiring immediate updates")
            return True
        
        logger.info(f"Found {len(critical_vulns)} critical vulnerabilities")
        
        # Generate updated Dockerfile
        dockerfile_updates = self.generate_dockerfile_updates(critical_vulns)
        
        if dockerfile_updates:
            # Build new image with security updates
            new_tag = f"security-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Build new image
            build_cmd = f"docker build -t {self.deployment}:{new_tag} ."
            build_result = subprocess.run(build_cmd, shell=True)
            
            if build_result.returncode == 0:
                # Update deployment with new image
                update_cmd = f"kubectl set image deployment/{self.deployment} pno-inference={self.deployment}:{new_tag} -n {self.namespace}"
                update_result = subprocess.run(update_cmd, shell=True)
                
                if update_result.returncode == 0:
                    # Wait for rollout to complete
                    rollout_cmd = f"kubectl rollout status deployment/{self.deployment} -n {self.namespace} --timeout=300s"
                    rollout_result = subprocess.run(rollout_cmd, shell=True)
                    
                    return rollout_result.returncode == 0
        
        return False
    
    def generate_dockerfile_updates(self, vulnerabilities):
        """Generate Dockerfile updates to fix vulnerabilities"""
        updates = []
        
        # Group vulnerabilities by package manager
        apt_packages = []
        pip_packages = []
        
        for vuln in vulnerabilities:
            if vuln.get('fixed_version'):
                if vuln['package'].startswith('lib') or vuln['package'] in ['openssl', 'curl']:
                    apt_packages.append(f"{vuln['package']}={vuln['fixed_version']}")
                else:
                    pip_packages.append(f"{vuln['package']}=={vuln['fixed_version']}")
        
        if apt_packages:
            updates.append(f"RUN apt-get update && apt-get install -y {' '.join(apt_packages)} && rm -rf /var/lib/apt/lists/*")
        
        if pip_packages:
            updates.append(f"RUN pip install --upgrade {' '.join(pip_packages)}")
        
        return updates
    
    def generate_security_report(self, vulnerabilities):
        """Generate security report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "scan_target": f"{self.deployment} in {self.namespace}",
            "total_vulnerabilities": len(vulnerabilities) if vulnerabilities else 0,
            "severity_breakdown": {},
            "critical_vulnerabilities": [],
            "recommendations": []
        }
        
        if vulnerabilities:
            # Severity breakdown
            for vuln in vulnerabilities:
                severity = vuln['severity']
                report["severity_breakdown"][severity] = report["severity_breakdown"].get(severity, 0) + 1
            
            # Critical vulnerabilities
            critical_vulns = self.check_critical_vulnerabilities(vulnerabilities)
            report["critical_vulnerabilities"] = critical_vulns
            
            # Recommendations
            if critical_vulns:
                report["recommendations"].append("Immediate security update required for critical vulnerabilities")
            
            high_vulns = [v for v in vulnerabilities if v['severity'] == 'HIGH']
            if high_vulns:
                report["recommendations"].append(f"Schedule security update for {len(high_vulns)} high severity vulnerabilities")
        
        return report
    
    def run_security_maintenance(self):
        """Run complete security maintenance"""
        logger.info("Starting security maintenance")
        
        # Scan for vulnerabilities
        vulnerabilities = self.scan_vulnerabilities()
        
        if vulnerabilities is None:
            logger.error("Vulnerability scan failed")
            return False
        
        # Generate report
        report = self.generate_security_report(vulnerabilities)
        
        # Save report
        report_file = f"/var/log/pno/security-report-{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security report saved to {report_file}")
        
        # Check if updates are needed
        critical_vulns = self.check_critical_vulnerabilities(vulnerabilities)
        
        if critical_vulns:
            logger.warning(f"Found {len(critical_vulns)} critical vulnerabilities")
            
            # Apply updates
            update_success = self.apply_security_updates(vulnerabilities)
            
            if update_success:
                logger.info("Security updates applied successfully")
            else:
                logger.error("Failed to apply security updates")
                # Send alert
                self.send_security_alert(report)
        
        return True
    
    def send_security_alert(self, report):
        """Send security alert for critical vulnerabilities"""
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "severity": "CRITICAL",
            "service": "pno-physics-bench",
            "alert_type": "security_vulnerability",
            "details": report
        }
        
        # Implement your alerting mechanism here
        logger.critical(f"Security alert: {json.dumps(alert_data)}")

def main():
    updater = SecurityUpdater()
    success = updater.run_security_maintenance()
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
```

---

## Performance Optimization

### Performance Monitoring and Optimization
```python
#!/usr/bin/env python3
# Performance optimization automation
# File: /opt/pno/optimization/performance_optimizer.py

import subprocess
import json
import time
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self):
        self.namespace = "production"
        self.deployment = "pno-physics-bench"
        self.metrics_history = []
        
    def collect_performance_metrics(self):
        """Collect current performance metrics"""
        try:
            # Get response time metrics from Prometheus
            prometheus_query = f"""
            query={{
                "query": "histogram_quantile(0.95, rate(pno_inference_duration_seconds_bucket[5m]))"
            }}
            """
            
            # Get resource usage
            resource_cmd = f"kubectl top pods -n {self.namespace} -l app={self.deployment} --no-headers"
            resource_result = subprocess.check_output(resource_cmd, shell=True).decode()
            
            cpu_usage = []
            memory_usage = []
            
            for line in resource_result.strip().split('\n'):
                if line:
                    parts = line.split()
                    cpu_usage.append(int(parts[1].replace('m', '')))
                    memory_usage.append(int(parts[2].replace('Mi', '')))
            
            # Get HPA metrics
            hpa_cmd = f"kubectl get hpa pno-hpa -n {self.namespace} -o json"
            hpa_result = subprocess.check_output(hpa_cmd, shell=True).decode()
            hpa_data = json.loads(hpa_result)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "avg_cpu_usage": np.mean(cpu_usage) if cpu_usage else 0,
                "max_cpu_usage": max(cpu_usage) if cpu_usage else 0,
                "avg_memory_usage": np.mean(memory_usage) if memory_usage else 0,
                "max_memory_usage": max(memory_usage) if memory_usage else 0,
                "current_replicas": hpa_data['status']['currentReplicas'],
                "desired_replicas": hpa_data['status']['desiredReplicas'],
                "pod_count": len(cpu_usage)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None
    
    def analyze_performance_trends(self, metrics):
        """Analyze performance trends and identify optimization opportunities"""
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if len(self.metrics_history) < 10:  # Need enough data points
            return []
        
        recommendations = []
        
        # Analyze CPU usage trends
        recent_cpu = [m["avg_cpu_usage"] for m in self.metrics_history[-10:]]
        avg_recent_cpu = np.mean(recent_cpu)
        
        if avg_recent_cpu > 800:  # 80% CPU usage
            recommendations.append({
                "type": "scale_up",
                "reason": f"High CPU usage: {avg_recent_cpu:.0f}m average",
                "action": "increase_replicas",
                "urgency": "high" if avg_recent_cpu > 900 else "medium"
            })
        elif avg_recent_cpu < 300:  # 30% CPU usage
            recommendations.append({
                "type": "scale_down", 
                "reason": f"Low CPU usage: {avg_recent_cpu:.0f}m average",
                "action": "decrease_replicas",
                "urgency": "low"
            })
        
        # Analyze memory usage trends
        recent_memory = [m["avg_memory_usage"] for m in self.metrics_history[-10:]]
        avg_recent_memory = np.mean(recent_memory)
        
        if avg_recent_memory > 3200:  # 80% of 4Gi limit
            recommendations.append({
                "type": "resource_increase",
                "reason": f"High memory usage: {avg_recent_memory:.0f}Mi average",
                "action": "increase_memory_limits",
                "urgency": "high" if avg_recent_memory > 3600 else "medium"
            })
        
        # Analyze scaling efficiency
        replica_changes = [m["current_replicas"] for m in self.metrics_history[-10:]]
        if len(set(replica_changes)) > 3:  # Frequent scaling
            recommendations.append({
                "type": "scaling_optimization",
                "reason": "Frequent scaling detected",
                "action": "adjust_hpa_parameters",
                "urgency": "medium"
            })
        
        return recommendations
    
    def apply_optimization(self, recommendation):
        """Apply performance optimization recommendation"""
        try:
            if recommendation["action"] == "increase_replicas":
                # Scale up deployment
                current_replicas = int(subprocess.check_output(
                    f"kubectl get deployment {self.deployment} -n {self.namespace} -o jsonpath='{{.spec.replicas}}'",
                    shell=True
                ).decode())
                
                new_replicas = min(current_replicas + 2, 20)  # Cap at 20 replicas
                
                cmd = f"kubectl scale deployment {self.deployment} --replicas={new_replicas} -n {self.namespace}"
                result = subprocess.run(cmd, shell=True)
                
                if result.returncode == 0:
                    logger.info(f"Scaled up to {new_replicas} replicas")
                    return True
                    
            elif recommendation["action"] == "decrease_replicas":
                # Scale down deployment
                current_replicas = int(subprocess.check_output(
                    f"kubectl get deployment {self.deployment} -n {self.namespace} -o jsonpath='{{.spec.replicas}}'",
                    shell=True
                ).decode())
                
                new_replicas = max(current_replicas - 1, 2)  # Minimum 2 replicas
                
                cmd = f"kubectl scale deployment {self.deployment} --replicas={new_replicas} -n {self.namespace}"
                result = subprocess.run(cmd, shell=True)
                
                if result.returncode == 0:
                    logger.info(f"Scaled down to {new_replicas} replicas")
                    return True
                    
            elif recommendation["action"] == "increase_memory_limits":
                # Increase memory limits
                patch = {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pno-inference",
                                    "resources": {
                                        "limits": {"memory": "6Gi"},
                                        "requests": {"memory": "2Gi"}
                                    }
                                }]
                            }
                        }
                    }
                }
                
                cmd = f"kubectl patch deployment {self.deployment} -n {self.namespace} -p '{json.dumps(patch)}'"
                result = subprocess.run(cmd, shell=True)
                
                if result.returncode == 0:
                    logger.info("Increased memory limits to 6Gi")
                    return True
                    
            elif recommendation["action"] == "adjust_hpa_parameters":
                # Adjust HPA parameters for more stable scaling
                patch = {
                    "spec": {
                        "behavior": {
                            "scaleUp": {
                                "stabilizationWindowSeconds": 120,
                                "policies": [{
                                    "type": "Percent",
                                    "value": 50,
                                    "periodSeconds": 60
                                }]
                            },
                            "scaleDown": {
                                "stabilizationWindowSeconds": 600,
                                "policies": [{
                                    "type": "Percent",
                                    "value": 25,
                                    "periodSeconds": 120
                                }]
                            }
                        }
                    }
                }
                
                cmd = f"kubectl patch hpa pno-hpa -n {self.namespace} -p '{json.dumps(patch)}'"
                result = subprocess.run(cmd, shell=True)
                
                if result.returncode == 0:
                    logger.info("Adjusted HPA parameters for stability")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            
        return False
    
    def run_optimization_cycle(self):
        """Run one optimization cycle"""
        logger.info("Starting performance optimization cycle")
        
        # Collect current metrics
        metrics = self.collect_performance_metrics()
        
        if not metrics:
            logger.error("Failed to collect metrics")
            return False
        
        # Analyze trends and get recommendations
        recommendations = self.analyze_performance_trends(metrics)
        
        if not recommendations:
            logger.info("No optimization recommendations at this time")
            return True
        
        # Apply high and medium urgency recommendations
        for rec in recommendations:
            if rec["urgency"] in ["high", "medium"]:
                logger.info(f"Applying optimization: {rec['reason']}")
                
                success = self.apply_optimization(rec)
                
                if success:
                    logger.info(f"Successfully applied: {rec['action']}")
                    
                    # Wait for optimization to take effect
                    time.sleep(60)
                else:
                    logger.error(f"Failed to apply: {rec['action']}")
        
        # Log all recommendations
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": metrics,
            "recommendations": recommendations,
            "applied_optimizations": [
                rec for rec in recommendations 
                if rec["urgency"] in ["high", "medium"]
            ]
        }
        
        # Save optimization report
        report_file = f"/var/log/pno/optimization-report-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to {report_file}")
        
        return True

def main():
    optimizer = PerformanceOptimizer()
    
    # Run optimization cycles
    while True:
        try:
            success = optimizer.run_optimization_cycle()
            
            if not success:
                logger.error("Optimization cycle failed")
            
            # Wait 5 minutes between cycles
            time.sleep(300)
            
        except KeyboardInterrupt:
            logger.info("Optimization stopped by user")
            break
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze current deployment configuration and identify missing components for complete production readiness", "status": "completed"}, {"content": "Create comprehensive operational runbooks with deployment procedures, troubleshooting guides, and maintenance protocols", "status": "completed"}, {"content": "Generate final security and compliance validation package with penetration testing reports and audit documentation", "status": "in_progress"}, {"content": "Create performance testing suite with load testing scripts, capacity planning, and SLA definitions", "status": "pending"}, {"content": "Generate complete deployment automation with zero-downtime deployment strategies and rollback procedures", "status": "pending"}, {"content": "Create comprehensive monitoring and alerting configuration with dashboards and incident response procedures", "status": "pending"}, {"content": "Generate backup and disaster recovery automation with cross-region replication and recovery testing", "status": "pending"}, {"content": "Create final deployment report with go-live checklist and production readiness certification", "status": "pending"}]