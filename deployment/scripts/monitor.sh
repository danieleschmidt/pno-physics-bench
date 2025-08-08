#!/bin/bash

NAMESPACE=${1:-default}

echo "PNO Physics Bench - Monitoring Dashboard"
echo "========================================"

# Pod status
echo "Pod Status:"
kubectl get pods -l app=pno-physics-bench -n $NAMESPACE

# Service status
echo -e "\nService Status:"
kubectl get services -l app=pno-physics-bench -n $NAMESPACE

# Resource usage
echo -e "\nResource Usage:"
kubectl top pods -l app=pno-physics-bench -n $NAMESPACE 2>/dev/null || echo "Metrics server not available"

# Recent logs
echo -e "\nRecent Logs:"
kubectl logs -l app=pno-physics-bench --tail=10 -n $NAMESPACE

# Health check
echo -e "\nHealth Check:"
SERVICE_IP=$(kubectl get service pno-physics-bench-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
curl -s http://$SERVICE_IP:8001/health || echo "Health check unavailable"
