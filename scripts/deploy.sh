#!/bin/bash

# SmartRAG deployment script
set -e

echo "🚀 Deploying SmartRAG..."

# Configuration
NAMESPACE=${NAMESPACE:-"smartrag"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
REGISTRY=${REGISTRY:-"docker.io"}
IMAGE_NAME=${IMAGE_NAME:-"smartrag"}

echo "📋 Deployment Configuration:"
echo "  Namespace: $NAMESPACE"
echo "  Environment: $ENVIRONMENT"
echo "  Image: $REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
fi

# Create namespace if it doesn't exist
echo "🏗️ Creating namespace if needed..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMap
echo "⚙️ Applying configuration..."
envsubst < k8s/configmap.yaml | kubectl apply -n $NAMESPACE -f -

# Apply Deployment
echo "🚀 Deploying application..."
envsubst < k8s/deployment.yaml | kubectl apply -n $NAMESPACE -f -

# Apply Service
echo "🌐 Creating service..."
kubectl apply -n $NAMESPACE -f k8s/service.yaml

# Apply HPA
echo "📈 Setting up autoscaling..."
kubectl apply -n $NAMESPACE -f k8s/hpa.yaml

# Apply Ingress (if exists)
if [ -f "k8s/ingress.yaml" ]; then
    echo "🌍 Setting up ingress..."
    kubectl apply -n $NAMESPACE -f k8s/ingress.yaml
fi

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/smartrag-api -n $NAMESPACE --timeout=300s

# Check if pods are running
echo "🔍 Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=smartrag-api

# Show service endpoints
echo "🌐 Service endpoints:"
kubectl get services -n $NAMESPACE

echo "✅ Deployment completed successfully!"

# Optional: Run health check
if [ "$HEALTH_CHECK" = "true" ]; then
    echo "🏥 Running health check..."
    
    # Get service URL
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        SERVICE_URL=$(minikube service smartrag-api --url -n $NAMESPACE)
    else
        # For cloud deployments, you might need to adjust this
        SERVICE_URL="http://localhost:8000"  # Assuming port-forward or LoadBalancer
    fi
    
    # Wait a bit for service to be ready
    sleep 10
    
    # Check health endpoint
    if curl -f "$SERVICE_URL/health" &> /dev/null; then
        echo "✅ Health check passed!"
    else
        echo "❌ Health check failed!"
        exit 1
    fi
fi