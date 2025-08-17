#!/bin/bash
# Automated Production Deployment Script for PNO Physics Bench

set -e

echo "🚀 Starting PNO Physics Bench Production Deployment..."

# Configuration
NAMESPACE="pno-production"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="latest"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $DOCKER_REGISTRY/pno-physics-bench:$IMAGE_TAG .
docker push $DOCKER_REGISTRY/pno-physics-bench:$IMAGE_TAG

# Apply Kubernetes manifests
echo "🎯 Deploying to Kubernetes..."

# Deploy Redis cache
if [[ "$REDIS_ENABLED" == "true" ]]; then
    echo "   • Deploying Redis cache..."
    kubectl apply -f redis-deployment.yaml -n $NAMESPACE
    kubectl apply -f redis-service.yaml -n $NAMESPACE
    kubectl apply -f redis-pvc.yaml -n $NAMESPACE
fi

# Deploy main application
echo "   • Deploying PNO Physics Bench..."
kubectl apply -f pno-deployment.yaml -n $NAMESPACE
kubectl apply -f pno-service.yaml -n $NAMESPACE
kubectl apply -f pno-pvc.yaml -n $NAMESPACE

# Deploy autoscaling
if [[ "$AUTO_SCALING" == "true" ]]; then
    echo "   • Configuring autoscaling..."
    kubectl apply -f pno-hpa.yaml -n $NAMESPACE
fi

# Deploy monitoring
if [[ "$MONITORING_ENABLED" == "true" ]]; then
    echo "   • Setting up monitoring..."
    kubectl apply -f prometheus-config.yaml -n $NAMESPACE
    kubectl apply -f grafana-deployment.yaml -n $NAMESPACE
fi

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/pno-physics-bench -n $NAMESPACE --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app=pno-physics-bench
kubectl get services -n $NAMESPACE

# Get service endpoint
EXTERNAL_IP=$(kubectl get service pno-physics-bench-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [[ -n "$EXTERNAL_IP" ]]; then
    echo "🌐 Service available at: http://$EXTERNAL_IP"
else
    echo "🔄 Waiting for external IP assignment..."
fi

# Run health check
echo "🏥 Running health check..."
kubectl port-forward -n $NAMESPACE service/pno-physics-bench-service 8080:80 &
PF_PID=$!
sleep 5

if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    kill $PF_PID
    exit 1
fi

kill $PF_PID

echo "🎉 Deployment completed successfully!"
echo "📊 Access metrics at: http://$EXTERNAL_IP/metrics"
echo "📚 Access docs at: http://$EXTERNAL_IP/docs"
