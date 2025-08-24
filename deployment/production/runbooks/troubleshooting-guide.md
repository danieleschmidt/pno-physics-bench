# PNO Physics Bench - Production Troubleshooting Guide

## Overview
Comprehensive troubleshooting guide for diagnosing and resolving issues in the PNO Physics Bench production environment.

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Diagnostic Commands](#diagnostic-commands)
3. [Common Issues](#common-issues)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Security Incident Response](#security-incident-response)
6. [Data and Model Issues](#data-and-model-issues)
7. [Infrastructure Problems](#infrastructure-problems)
8. [Escalation Procedures](#escalation-procedures)

## Quick Reference

### Emergency Contacts
- **Production Support**: +1-555-PROD-OPS (776-3677)
- **Security Team**: +1-555-SEC-TEAM (732-8326)
- **Infrastructure Team**: +1-555-INFRA-OPS (463-7267)
- **Slack Channels**: #pno-production-alerts, #incident-response

### Critical Commands
```bash
# Service status
kubectl get pods -n production -l app=pno-physics-bench

# Service logs
kubectl logs -f deployment/pno-physics-bench -n production

# Resource usage
kubectl top pods -n production

# Emergency rollback
kubectl rollout undo deployment/pno-physics-bench -n production

# Scale emergency replicas
kubectl scale deployment pno-physics-bench --replicas=5 -n production
```

### Service Endpoints
- **Health Check**: https://api.pno-physics-bench.com/health
- **Readiness**: https://api.pno-physics-bench.com/ready
- **Metrics**: https://api.pno-physics-bench.com/metrics
- **Grafana Dashboard**: https://grafana.company.com/d/pno-production

## Diagnostic Commands

### Application Diagnostics
```bash
# Check pod status and health
kubectl get pods -n production -l app=pno-physics-bench -o wide

# Get detailed pod information
kubectl describe pod <pod-name> -n production

# Check application logs
kubectl logs <pod-name> -n production --tail=100

# Stream live logs
kubectl logs -f deployment/pno-physics-bench -n production

# Check previous container logs (if pod restarted)
kubectl logs <pod-name> -n production --previous

# Execute commands in container
kubectl exec -it <pod-name> -n production -- bash

# Check resource usage
kubectl top pod <pod-name> -n production --containers
```

### Service and Network Diagnostics
```bash
# Check service configuration
kubectl describe service pno-service -n production

# Check service endpoints
kubectl get endpoints pno-service -n production

# Check ingress configuration
kubectl describe ingress pno-ingress -n production

# Check network policies
kubectl get networkpolicy -n production -o yaml

# Test internal connectivity
kubectl run debug-pod --image=busybox --rm -it -- sh
# Then inside: wget -qO- http://pno-service.production.svc.cluster.local/health
```

### Cluster and Node Diagnostics
```bash
# Check cluster health
kubectl cluster-info
kubectl get nodes -o wide

# Check node resource usage
kubectl describe node <node-name>
kubectl top nodes

# Check system pods
kubectl get pods -n kube-system

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | head -20
```

### Storage and Configuration Diagnostics
```bash
# Check persistent volumes
kubectl get pv,pvc -n production

# Check configmaps and secrets
kubectl get configmap,secret -n production

# Check volume mounts
kubectl describe pod <pod-name> -n production | grep -A 10 "Mounts:"

# Check storage class
kubectl get storageclass
```

## Common Issues

### Issue: Pods Stuck in Pending State

**Symptoms**:
- Pods remain in "Pending" status
- No containers are running

**Diagnosis**:
```bash
# Check pod details
kubectl describe pod <pod-name> -n production

# Check node resources
kubectl top nodes
kubectl describe nodes

# Check resource quotas
kubectl describe quota -n production
```

**Common Causes & Solutions**:

1. **Insufficient Resources**:
   ```bash
   # Check if nodes have enough CPU/memory
   kubectl top nodes
   
   # Solution: Scale cluster or reduce resource requests
   kubectl scale deployment pno-physics-bench --replicas=2 -n production
   ```

2. **Node Selector Issues**:
   ```bash
   # Check if nodes have required labels
   kubectl get nodes --show-labels | grep ml-optimized
   
   # Solution: Add label to nodes or remove nodeSelector
   kubectl label nodes <node-name> node-type=ml-optimized
   ```

3. **Image Pull Issues**:
   ```bash
   # Check image pull secrets
   kubectl get secrets -n production
   
   # Solution: Verify registry access and credentials
   kubectl create secret docker-registry regcred --docker-server=ghcr.io --docker-username=<user> --docker-password=<token>
   ```

### Issue: High Error Rate

**Symptoms**:
- HTTP 5xx responses increasing
- Application errors in logs
- Failed health checks

**Diagnosis**:
```bash
# Check error rate metrics
curl https://api.pno-physics-bench.com/metrics | grep error

# Check application logs for errors
kubectl logs deployment/pno-physics-bench -n production | grep -i error

# Check health endpoint
curl -v https://api.pno-physics-bench.com/health
```

**Common Causes & Solutions**:

1. **Model Loading Issues**:
   ```bash
   # Check if model files are accessible
   kubectl exec deployment/pno-physics-bench -n production -- ls -la /app/models/
   
   # Solution: Verify model storage mount
   kubectl describe pvc pno-model-storage -n production
   ```

2. **Resource Constraints**:
   ```bash
   # Check if pods are being killed due to resource limits
   kubectl describe pod <pod-name> -n production | grep -i killed
   
   # Solution: Increase resource limits
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","resources":{"limits":{"memory":"6Gi"}}}]}}}}'
   ```

3. **Configuration Issues**:
   ```bash
   # Check environment variables
   kubectl exec deployment/pno-physics-bench -n production -- env | grep PNO
   
   # Solution: Update configuration
   kubectl edit configmap pno-config -n production
   ```

### Issue: High Latency

**Symptoms**:
- API responses taking > 500ms
- Timeout errors from clients
- Poor user experience

**Diagnosis**:
```bash
# Check response time metrics
curl https://api.pno-physics-bench.com/metrics | grep duration

# Test direct API call timing
time curl https://api.pno-physics-bench.com/predict

# Check resource utilization
kubectl top pods -n production -l app=pno-physics-bench
```

**Common Causes & Solutions**:

1. **Insufficient Replicas**:
   ```bash
   # Check current replica count
   kubectl get deployment pno-physics-bench -n production
   
   # Solution: Scale up replicas
   kubectl scale deployment pno-physics-bench --replicas=5 -n production
   ```

2. **CPU Throttling**:
   ```bash
   # Check CPU usage vs limits
   kubectl top pods -n production -l app=pno-physics-bench --containers
   
   # Solution: Increase CPU limits
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","resources":{"limits":{"cpu":"3000m"}}}]}}}}'
   ```

3. **Network Issues**:
   ```bash
   # Test internal network latency
   kubectl run network-test --image=busybox --rm -it -- sh
   # Inside pod: time wget -qO- http://pno-service.production.svc.cluster.local/health
   
   # Solution: Check network policies and ingress configuration
   kubectl describe ingress pno-ingress -n production
   ```

### Issue: Pod Restart Loop

**Symptoms**:
- Pods continuously restarting
- CrashLoopBackOff status
- Application unavailable

**Diagnosis**:
```bash
# Check restart count
kubectl get pods -n production -l app=pno-physics-bench

# Check exit codes
kubectl describe pod <pod-name> -n production

# Check logs from failed container
kubectl logs <pod-name> -n production --previous
```

**Common Causes & Solutions**:

1. **Application Startup Failure**:
   ```bash
   # Check application logs for startup errors
   kubectl logs <pod-name> -n production --previous | head -50
   
   # Solution: Fix configuration or dependencies
   kubectl edit configmap pno-config -n production
   ```

2. **Health Check Failure**:
   ```bash
   # Check liveness/readiness probe configuration
   kubectl describe pod <pod-name> -n production | grep -A 10 "Liveness:"
   
   # Solution: Adjust probe settings
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","livenessProbe":{"initialDelaySeconds":60}}]}}}}'
   ```

3. **Resource Limits**:
   ```bash
   # Check if pod is killed due to OOMKilled
   kubectl describe pod <pod-name> -n production | grep -i oom
   
   # Solution: Increase memory limits
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","resources":{"limits":{"memory":"6Gi"}}}]}}}}'
   ```

## Performance Troubleshooting

### Latency Analysis

**Step 1: Identify Bottlenecks**
```bash
# Check P95 latency metrics
curl https://api.pno-physics-bench.com/metrics | grep -E "(inference_duration|request_duration)"

# Analyze request traces
kubectl logs deployment/pno-physics-bench -n production | grep "request_id" | tail -10
```

**Step 2: Resource Analysis**
```bash
# Check resource utilization trends
kubectl top pods -n production -l app=pno-physics-bench --sort-by=cpu
kubectl top pods -n production -l app=pno-physics-bench --sort-by=memory

# Check HPA status
kubectl describe hpa pno-hpa -n production
```

**Step 3: Optimization Actions**

1. **Scale Horizontally**:
   ```bash
   # Increase max replicas
   kubectl patch hpa pno-hpa -n production -p '{"spec":{"maxReplicas":10}}'
   
   # Manually scale for immediate relief
   kubectl scale deployment pno-physics-bench --replicas=8 -n production
   ```

2. **Optimize Resource Allocation**:
   ```bash
   # Increase resource limits
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","resources":{"limits":{"cpu":"4000m","memory":"8Gi"},"requests":{"cpu":"1000m","memory":"2Gi"}}}]}}}}'
   ```

3. **Enable Performance Features**:
   ```bash
   # Enable performance optimization
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","env":[{"name":"OPTIMIZATION_LEVEL","value":"aggressive"}]}]}}}}'
   ```

### Memory Issues

**Diagnosis**:
```bash
# Check memory usage patterns
kubectl top pods -n production -l app=pno-physics-bench --sort-by=memory

# Check for memory leaks
kubectl exec deployment/pno-physics-bench -n production -- python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**Solutions**:

1. **Memory Leak Detection**:
   ```bash
   # Enable memory profiling
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","env":[{"name":"MEMORY_PROFILING","value":"true"}]}]}}}}'
   
   # Collect memory dump
   kubectl exec deployment/pno-physics-bench -n production -- python scripts/memory_analysis.py
   ```

2. **Memory Optimization**:
   ```bash
   # Enable model quantization
   kubectl patch configmap pno-config -n production -p '{"data":{"model_quantization":"true"}}'
   
   # Enable gradient checkpointing
   kubectl patch configmap pno-config -n production -p '{"data":{"gradient_checkpointing":"true"}}'
   ```

### CPU Performance

**Diagnosis**:
```bash
# Check CPU throttling
kubectl top pods -n production -l app=pno-physics-bench --sort-by=cpu

# Check CPU utilization history
curl https://prometheus.company.com/api/v1/query?query=rate(container_cpu_usage_seconds_total{pod=~"pno-physics-bench.*"}[5m])
```

**Solutions**:

1. **CPU Optimization**:
   ```bash
   # Enable CPU optimization flags
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","env":[{"name":"OMP_NUM_THREADS","value":"4"},{"name":"TORCH_THREADS","value":"4"}]}]}}}}'
   ```

2. **GPU Acceleration** (if available):
   ```bash
   # Add GPU resources
   kubectl patch deployment pno-physics-bench -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"pno-inference","resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}}}'
   ```

## Security Incident Response

### Security Alert Response

**Step 1: Immediate Assessment**
```bash
# Check for suspicious activity
kubectl logs deployment/pno-physics-bench -n production | grep -E "(failed|unauthorized|403|401)"

# Check running processes in containers
kubectl exec deployment/pno-physics-bench -n production -- ps aux

# Check network connections
kubectl exec deployment/pno-physics-bench -n production -- netstat -tulpn
```

**Step 2: Containment**
```bash
# Isolate affected pods
kubectl label pod <suspicious-pod> quarantine=true -n production

# Update network policy to block suspicious traffic
kubectl apply -f security/emergency-network-policy.yaml

# Scale down if necessary
kubectl scale deployment pno-physics-bench --replicas=1 -n production
```

**Step 3: Investigation**
```bash
# Collect forensic data
kubectl logs <suspicious-pod> -n production > incident-logs-$(date +%Y%m%d-%H%M).log

# Check file system changes
kubectl exec <suspicious-pod> -n production -- find /app -type f -newermt '1 hour ago'

# Export pod configuration
kubectl get pod <suspicious-pod> -n production -o yaml > incident-pod-config.yaml
```

**Step 4: Recovery**
```bash
# Deploy clean version
kubectl rollout restart deployment/pno-physics-bench -n production

# Update security policies
kubectl apply -f security/updated-security-policies.yaml

# Verify clean deployment
python security/verify_deployment_security.py
```

### Vulnerability Response

**Critical Vulnerability Process**:

1. **Assessment** (Within 2 hours):
   ```bash
   # Run security scan
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/app aquasec/trivy image pno-physics-bench:latest
   
   # Check affected components
   kubectl exec deployment/pno-physics-bench -n production -- pip list | grep <vulnerable-package>
   ```

2. **Patching** (Within 24 hours):
   ```bash
   # Build patched image
   docker build -t pno-physics-bench:security-patch-$(date +%Y%m%d) .
   
   # Deploy security update
   kubectl set image deployment/pno-physics-bench pno-inference=pno-physics-bench:security-patch-$(date +%Y%m%d) -n production
   ```

3. **Validation** (Within 4 hours of patch):
   ```bash
   # Verify patch applied
   kubectl exec deployment/pno-physics-bench -n production -- pip show <patched-package>
   
   # Run security validation
   python security/validate_security_patches.py
   ```

## Data and Model Issues

### Model Loading Failures

**Diagnosis**:
```bash
# Check model storage
kubectl describe pvc pno-model-storage -n production

# Check model files
kubectl exec deployment/pno-physics-bench -n production -- ls -la /app/models/

# Check model loading logs
kubectl logs deployment/pno-physics-bench -n production | grep -i "model"
```

**Solutions**:

1. **Storage Issues**:
   ```bash
   # Check storage capacity
   kubectl exec deployment/pno-physics-bench -n production -- df -h /app/models/
   
   # Remount storage if needed
   kubectl delete pod -l app=pno-physics-bench -n production
   ```

2. **Model Corruption**:
   ```bash
   # Verify model checksums
   kubectl exec deployment/pno-physics-bench -n production -- python scripts/verify_model_integrity.py
   
   # Restore from backup
   kubectl exec deployment/pno-physics-bench -n production -- cp /app/backups/model.pth /app/models/
   ```

### Inference Failures

**Diagnosis**:
```bash
# Check inference error patterns
kubectl logs deployment/pno-physics-bench -n production | grep -i "inference.*error"

# Test inference endpoint
curl -X POST https://api.pno-physics-bench.com/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "test_data"}'
```

**Solutions**:

1. **Input Validation Issues**:
   ```bash
   # Check input validation logs
   kubectl logs deployment/pno-physics-bench -n production | grep -i "validation"
   
   # Update validation rules
   kubectl patch configmap pno-config -n production -p '{"data":{"input_validation":"relaxed"}}'
   ```

2. **Model Compatibility**:
   ```bash
   # Check model version compatibility
   kubectl exec deployment/pno-physics-bench -n production -- python -c "import torch; print(torch.__version__)"
   
   # Rollback to compatible version
   kubectl rollout undo deployment/pno-physics-bench -n production
   ```

## Infrastructure Problems

### Kubernetes Cluster Issues

**Node Problems**:
```bash
# Check node status
kubectl get nodes -o wide
kubectl describe node <node-name>

# Check node resource usage
kubectl top nodes

# Check node events
kubectl get events --field-selector involvedObject.kind=Node
```

**Storage Issues**:
```bash
# Check persistent volume status
kubectl get pv,pvc --all-namespaces

# Check storage classes
kubectl get storageclass

# Check volume mount issues
kubectl describe pod <pod-name> -n production | grep -A 5 "Events:"
```

**Network Issues**:
```bash
# Check CNI status
kubectl get pods -n kube-system | grep -E "(calico|flannel|weave)"

# Check DNS resolution
kubectl run debug-pod --image=busybox --rm -it -- nslookup pno-service.production.svc.cluster.local

# Check ingress controller
kubectl get pods -n ingress-nginx
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

### Load Balancer Issues

**Diagnosis**:
```bash
# Check ingress status
kubectl get ingress -n production -o wide
kubectl describe ingress pno-ingress -n production

# Check load balancer service
kubectl get service pno-service -n production -o wide

# Test external connectivity
curl -v https://api.pno-physics-bench.com/health
```

**Solutions**:

1. **DNS Issues**:
   ```bash
   # Check DNS records
   nslookup api.pno-physics-bench.com
   
   # Update DNS if needed (external DNS management)
   ```

2. **Certificate Issues**:
   ```bash
   # Check certificate status
   kubectl get certificate -n production
   kubectl describe certificate pno-tls-cert -n production
   
   # Renew certificate if needed
   kubectl delete certificate pno-tls-cert -n production
   kubectl apply -f certificates/pno-tls-cert.yaml
   ```

## Escalation Procedures

### Escalation Matrix

**Level 1 - Operations Team** (Response: 15 minutes):
- Service degradation
- Performance issues
- Non-critical errors

**Level 2 - Engineering Team** (Response: 30 minutes):
- Service outages
- Security alerts
- Data integrity issues

**Level 3 - Architecture Team** (Response: 1 hour):
- Infrastructure failures
- Design-level issues
- Major security breaches

**Level 4 - Executive Team** (Response: 2 hours):
- Business-critical outages
- Data breaches
- Compliance violations

### Escalation Process

1. **Initial Response**:
   ```bash
   # Document issue
   echo "Issue: $(date) - Service degradation observed" >> /var/log/incidents/current.log
   
   # Attempt immediate resolution
   kubectl rollout restart deployment/pno-physics-bench -n production
   
   # Monitor for 15 minutes
   ```

2. **Level 1 Escalation**:
   - Create incident ticket
   - Notify operations Slack channel
   - Begin detailed troubleshooting

3. **Level 2 Escalation**:
   - Page engineering on-call
   - Join incident bridge
   - Collect diagnostic data

4. **Level 3 Escalation**:
   - Page architecture team
   - Consider rollback to last known good
   - Activate disaster recovery if needed

### Communication Templates

**Initial Alert**:
```
INCIDENT: PNO Physics Bench Production Issue
Time: [TIMESTAMP]
Severity: [P1/P2/P3/P4]
Status: Investigating
Impact: [USER IMPACT DESCRIPTION]
Next Update: [TIME]
```

**Status Update**:
```
INCIDENT UPDATE: PNO Physics Bench
Time: [TIMESTAMP]
Status: [Investigating/Identified/Monitoring/Resolved]
Actions Taken: [ACTIONS]
Current Impact: [CURRENT STATE]
Next Update: [TIME]
```

**Resolution Notice**:
```
INCIDENT RESOLVED: PNO Physics Bench
Resolution Time: [TIMESTAMP]
Root Cause: [BRIEF DESCRIPTION]
Actions Taken: [RESOLUTION STEPS]
Follow-up: [POST-INCIDENT ACTIONS]
```

---

## Appendix

### Useful Commands Reference
```bash
# Quick health check
kubectl get pods -n production -l app=pno-physics-bench && curl -s https://api.pno-physics-bench.com/health

# Resource usage summary
kubectl top pods -n production --sort-by=memory

# Event monitoring
kubectl get events -n production --watch

# Log streaming
kubectl logs -f deployment/pno-physics-bench -n production

# Emergency scale
kubectl scale deployment pno-physics-bench --replicas=1 -n production
```

### Log Analysis Patterns
```bash
# Error patterns
kubectl logs deployment/pno-physics-bench -n production | grep -E "(ERROR|CRITICAL|FATAL)"

# Performance patterns
kubectl logs deployment/pno-physics-bench -n production | grep -E "(slow|timeout|latency)"

# Security patterns
kubectl logs deployment/pno-physics-bench -n production | grep -E "(failed.*auth|unauthorized|403|401)"
```

### Contact Information
- **Emergency Hotline**: +1-555-EMERGENCY
- **Operations Desk**: ops@company.com
- **Security Team**: security@company.com
- **Engineering Team**: engineering@company.com

---
**Document Version**: 1.0.0  
**Last Updated**: 2025-08-23  
**Next Review**: 2025-11-23