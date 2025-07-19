#!/bin/bash

# Script to run load_airgapped_experiments_to_db.py in tao-api-app container
# This script handles running the air-gapped experiment loader within either:
# - Kubernetes pods (using kubectl)
# - Docker Compose containers (using docker exec)

set -e  # Exit on any error

# Default values
DEPLOYMENT_TYPE="kubernetes"  # kubernetes or docker-compose
NAMESPACE="default"
POD_PREFIX="tao-api-app"
CONTAINER_NAME="tao_api_app"  # Docker Compose service name
COMPOSE_FILE="docker-compose.yml"  # Docker Compose file path
JSON_FILE_PATH="index.json"  # Fixed path under LOCAL_MODEL_REGISTRY
DRY_RUN=false
VERBOSE=false
USE_CLOUD_STORAGE=true
CLOUD_TYPE="seaweedfs"
BUCKET_NAME="tao-storage"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Load air-gapped experiment metadata from index.json file in cloud storage to database via tao-api-app container.
The script always looks for 'index.json' under the LOCAL_MODEL_REGISTRY folder in cloud storage.
Supports both Kubernetes pods and Docker Compose containers.

OPTIONS:
    -t, --deployment-type TYPE      Deployment type: 'kubernetes' or 'docker-compose' (default: kubernetes)
    -n, --namespace NAMESPACE       Kubernetes namespace (default: default)
    -p, --pod-prefix PREFIX         Pod name prefix to search for (default: tao-api-app)
    -c, --container-name NAME       Docker Compose service name (default: tao_api_app)
    -f, --compose-file FILE         Docker Compose file path (optional, default: docker-compose.yml)
    -d, --dry-run                   Validate data without writing to database
    -v, --verbose                   Enable verbose logging
    --cloud-type TYPE               Cloud storage type (default: seaweedfs)
    --bucket-name NAME              Cloud storage bucket name (default: tao-storage)
    --endpoint-url URL              Cloud storage endpoint URL (overrides env var)
    --access-key KEY                Cloud storage access key (overrides env var)
    --secret-key KEY                Cloud storage secret key (overrides env var)
    --region REGION                 Cloud storage region
    -h, --help                      Show this help message

ENVIRONMENT VARIABLES:
    The following environment variables should be set in the container:
    - LOCAL_MODEL_REGISTRY          Base path in cloud storage for models
    - SEAWEEDFS_S3_ENDPOINT        SeaweedFS S3 endpoint URL
    - SEAWEEDFS_ACCESS_KEY         SeaweedFS access key
    - SEAWEEDFS_SECRET_KEY         SeaweedFS secret key

EXAMPLES:
    # Kubernetes examples
    $0 --dry-run
    $0 --namespace tao-system --verbose
    $0 --deployment-type kubernetes --pod-prefix custom-tao-app

    # Docker Compose examples
    $0 --deployment-type docker-compose --dry-run
    $0 -t docker-compose --container-name tao_api_app --verbose
    $0 -t docker-compose --compose-file /path/to/docker-compose.yml

    # Custom cloud storage settings (works with both deployment types)
    $0 --cloud-type seaweedfs --bucket-name my-bucket

EOF
}

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to find the tao-api-app pod (Kubernetes)
find_pod() {
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$POD_PREFIX" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "$pod_name" ]; then
        # Try alternative label selectors
        pod_name=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running -o jsonpath='{.items[?(@.metadata.name=="'$POD_PREFIX'*")].metadata.name}' 2>/dev/null | awk '{print $1}')
    fi
    
    if [ -z "$pod_name" ]; then
        # Try finding by name pattern
        pod_name=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running -o name 2>/dev/null | grep "$POD_PREFIX" | head -1 | cut -d'/' -f2)
    fi
    
    echo "$pod_name"
}

# Function to find the tao-api-app container (Docker Compose)
find_container() {
    # First try to find running container with exact name (most reliable for standardized setups)
    local container_id=$(docker ps --format "table {{.ID}}\t{{.Names}}" | grep "^[a-f0-9]*[[:space:]]*${CONTAINER_NAME}[[:space:]]*$" | awk '{print $1}' | head -1)
    
    if [ -z "$container_id" ]; then
        # Try finding by name pattern (allows partial matches)
        container_id=$(docker ps --format "table {{.ID}}\t{{.Names}}" | grep "$CONTAINER_NAME" | awk '{print $1}' | head -1)
    fi
    
    if [ -z "$container_id" ]; then
        # Try using Docker Compose (if compose file exists and is valid)
        if [ -f "$COMPOSE_FILE" ]; then
            container_id=$(docker compose -f "$COMPOSE_FILE" ps -q "$CONTAINER_NAME" 2>/dev/null)
        fi
    fi
    
    if [ -z "$container_id" ]; then
        # Final fallback: find by image name pattern
        container_id=$(docker ps --format "table {{.ID}}\t{{.Image}}" | grep -i tao | grep -i api | awk '{print $1}' | head -1)
    fi
    
    echo "$container_id"
}

# Function to check if required environment variables are set in pod (Kubernetes)
check_pod_env() {
    local pod_name="$1"
    log "Checking required environment variables in pod $pod_name..."
    
    # Check LOCAL_MODEL_REGISTRY
    local local_model_registry=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- env | grep LOCAL_MODEL_REGISTRY || true)
    if [ -z "$local_model_registry" ]; then
        log_error "LOCAL_MODEL_REGISTRY environment variable is not set in pod $pod_name"
        return 1
    fi
    log "Found: $local_model_registry"
    
    # Check SeaweedFS environment variables
    local seaweedfs_endpoint=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- env | grep SEAWEEDFS_S3_ENDPOINT || true)
    local seaweedfs_access=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- env | grep SEAWEEDFS_ACCESS_KEY || true)
    local seaweedfs_secret=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- env | grep SEAWEEDFS_SECRET_KEY || true)
    
    if [ -z "$seaweedfs_endpoint" ] || [ -z "$seaweedfs_access" ] || [ -z "$seaweedfs_secret" ]; then
        log_warning "Some SeaweedFS environment variables are not set in pod. You may need to provide them via command line arguments."
        log "SEAWEEDFS_S3_ENDPOINT: ${seaweedfs_endpoint:-'NOT SET'}"
        log "SEAWEEDFS_ACCESS_KEY: ${seaweedfs_access:-'NOT SET'}"
        log "SEAWEEDFS_SECRET_KEY: ${seaweedfs_secret:-'NOT SET'}"
    else
        log "SeaweedFS environment variables are configured"
    fi
    
    return 0
}

# Function to check if required environment variables are set in container (Docker Compose)
check_container_env() {
    local container_id="$1"
    log "Checking required environment variables in container $container_id..."
    
    # Check LOCAL_MODEL_REGISTRY
    local local_model_registry=$(docker exec "$container_id" env | grep LOCAL_MODEL_REGISTRY || true)
    if [ -z "$local_model_registry" ]; then
        log_error "LOCAL_MODEL_REGISTRY environment variable is not set in container $container_id"
        return 1
    fi
    log "Found: $local_model_registry"
    
    # Check SeaweedFS environment variables
    local seaweedfs_endpoint=$(docker exec "$container_id" env | grep SEAWEEDFS_S3_ENDPOINT || true)
    local seaweedfs_access=$(docker exec "$container_id" env | grep SEAWEEDFS_ACCESS_KEY || true)
    local seaweedfs_secret=$(docker exec "$container_id" env | grep SEAWEEDFS_SECRET_KEY || true)
    
    if [ -z "$seaweedfs_endpoint" ] || [ -z "$seaweedfs_access" ] || [ -z "$seaweedfs_secret" ]; then
        log_warning "Some SeaweedFS environment variables are not set in container. You may need to provide them via command line arguments."
        log "SEAWEEDFS_S3_ENDPOINT: ${seaweedfs_endpoint:-'NOT SET'}"
        log "SEAWEEDFS_ACCESS_KEY: ${seaweedfs_access:-'NOT SET'}"
        log "SEAWEEDFS_SECRET_KEY: ${seaweedfs_secret:-'NOT SET'}"
    else
        log "SeaweedFS environment variables are configured"
    fi
    
    return 0
}

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--deployment-type)
            DEPLOYMENT_TYPE="$2"
            if [[ "$DEPLOYMENT_TYPE" != "kubernetes" && "$DEPLOYMENT_TYPE" != "docker-compose" ]]; then
                log_error "Invalid deployment type: $DEPLOYMENT_TYPE. Must be 'kubernetes' or 'docker-compose'"
                exit 1
            fi
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -p|--pod-prefix)
            POD_PREFIX="$2"
            shift 2
            ;;
        -c|--container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -f|--compose-file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --cloud-type)
            CLOUD_TYPE="$2"
            EXTRA_ARGS+=(--cloud-type "$2")
            shift 2
            ;;
        --bucket-name)
            BUCKET_NAME="$2"
            EXTRA_ARGS+=(--bucket-name "$2")
            shift 2
            ;;
        --endpoint-url)
            EXTRA_ARGS+=(--endpoint-url "$2")
            shift 2
            ;;
        --access-key)
            EXTRA_ARGS+=(--access-key "$2")
            shift 2
            ;;
        --secret-key)
            EXTRA_ARGS+=(--secret-key "$2")
            shift 2
            ;;
        --region)
            EXTRA_ARGS+=(--region "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            log_error "Unexpected positional argument: $1. This script does not accept positional arguments."
            log_error "The JSON file is always 'index.json' under LOCAL_MODEL_REGISTRY."
            usage
            exit 1
            ;;
    esac
done

# Check required tools and find container/pod based on deployment type
if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl command not found. Please install kubectl and configure access to your cluster."
        exit 1
    fi

    # Find the pod
    log "Searching for $POD_PREFIX pod in namespace $NAMESPACE..."
    CONTAINER_OR_POD=$(find_pod)

    if [ -z "$CONTAINER_OR_POD" ]; then
        log_error "No running $POD_PREFIX pod found in namespace $NAMESPACE"
        log "Available pods in namespace $NAMESPACE:"
        kubectl get pods -n "$NAMESPACE" 2>/dev/null || log_error "Failed to list pods. Check if namespace exists and you have access."
        exit 1
    fi

    log "Found pod: $CONTAINER_OR_POD"

    # Check pod environment
    if ! check_pod_env "$CONTAINER_OR_POD"; then
        log_error "Pod environment check failed"
        exit 1
    fi

elif [ "$DEPLOYMENT_TYPE" = "docker-compose" ]; then
    # Check if docker and docker compose are available
    if ! command -v docker &> /dev/null; then
        log_error "docker command not found. Please install Docker."
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        log_error "docker compose not available. Please install Docker Compose or use legacy docker-compose."
        exit 1
    fi

    # Find the container
    log "Searching for $CONTAINER_NAME container using compose file $COMPOSE_FILE..."
    CONTAINER_OR_POD=$(find_container)

    if [ -z "$CONTAINER_OR_POD" ]; then
        log_error "No running $CONTAINER_NAME container found"
        log "Available containers:"
        docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" 2>/dev/null || log_error "Failed to list containers."
        if [ -f "$COMPOSE_FILE" ]; then
            log "Available compose services:"
            docker compose -f "$COMPOSE_FILE" ps 2>/dev/null || log_warning "Could not list compose services from: $COMPOSE_FILE"
        else
            log_warning "Compose file not found: $COMPOSE_FILE (this is optional if container name is standardized)"
        fi
        exit 1
    fi

    log "Found container: $CONTAINER_OR_POD"

    # Check container environment
    if ! check_container_env "$CONTAINER_OR_POD"; then
        log_error "Container environment check failed"
        exit 1
    fi

else
    log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    exit 1
fi

# Build the command to run in the pod
PYTHON_CMD="python3 -m nvidia_tao_core.microservices.load_airgapped_experiments_to_db "
PYTHON_CMD+=" --use-cloud-storage"
PYTHON_CMD+=" --cloud-type \"$CLOUD_TYPE\""
PYTHON_CMD+=" --bucket-name \"$BUCKET_NAME\""

if [ "$DRY_RUN" = true ]; then
    PYTHON_CMD+=" --dry-run"
fi

if [ "$VERBOSE" = true ]; then
    PYTHON_CMD+=" --verbose"
fi

# Add extra arguments
for arg in "${EXTRA_ARGS[@]}"; do
    PYTHON_CMD+=" \"$arg\""
done

log "Executing command in $DEPLOYMENT_TYPE container/pod $CONTAINER_OR_POD:"
log "$PYTHON_CMD"

# Execute the command based on deployment type
log "Starting air-gapped experiment import from index.json..."
if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
    if kubectl exec -n "$NAMESPACE" "$CONTAINER_OR_POD" -- bash -c "$PYTHON_CMD"; then
        log "Air-gapped experiment import completed successfully!"
    else
        log_error "Air-gapped experiment import failed!"
        exit 1
    fi
elif [ "$DEPLOYMENT_TYPE" = "docker-compose" ]; then
    if docker exec "$CONTAINER_OR_POD" bash -c "$PYTHON_CMD"; then
        log "Air-gapped experiment import completed successfully!"
    else
        log_error "Air-gapped experiment import failed!"
        exit 1
    fi
fi 