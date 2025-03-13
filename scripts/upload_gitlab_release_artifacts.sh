#!/bin/bash
set -e

# Script to upload third-party wheels as GitLab release artifacts
# Expected environment variables:
# - GITLAB_TOKEN: API token for GitLab
# - GITLAB_PROJECT_ID: GitLab project ID
# - TAG_NAME: Tag name for the release
# - RELEASE_NAME: Name of the release
# - RELEASE_DESCRIPTION: Description for the release
# - WHEELS_DIR: Directory containing the wheel files

# Function to check if required environment variables are set
check_required_vars() {
  local missing_vars=0
  
  if [ -z "${GITLAB_TOKEN}" ]; then
    echo "ERROR: GITLAB_TOKEN environment variable is not set"
    missing_vars=1
  fi
  
  if [ -z "${GITLAB_PROJECT_ID}" ]; then
    echo "ERROR: GITLAB_PROJECT_ID environment variable is not set"
    missing_vars=1
  fi
  
  if [ -z "${TAG_NAME}" ]; then
    echo "ERROR: TAG_NAME environment variable is not set"
    missing_vars=1
  fi
  
  if [ -z "${WHEELS_DIR}" ]; then
    echo "WARNING: WHEELS_DIR environment variable is not set, defaulting to 'third_party_wheels'"
    export WHEELS_DIR="third_party_wheels"
  fi

  if [ -z "${RELEASE_NAME}" ]; then
    echo "WARNING: RELEASE_NAME environment variable is not set, defaulting to 'Release ${TAG_NAME}'"
    export RELEASE_NAME="Release ${TAG_NAME}"
  fi
  
  if [ -z "${RELEASE_DESCRIPTION}" ]; then
    echo "WARNING: RELEASE_DESCRIPTION environment variable is not set, defaulting to 'Third-party wheels for ${TAG_NAME}'"
    export RELEASE_DESCRIPTION="Third-party wheels for ${TAG_NAME}"
  fi
  
  if [ $missing_vars -eq 1 ]; then
    echo "ERROR: Required environment variables are missing. Exiting."
    exit 1
  fi
}

# Function to execute a curl command with retries
curl_with_retry() {
  local url=$1
  local method=${2:-"GET"}
  local data=${3:-""}
  local upload_file=${4:-""}
  local max_attempts=${5:-3}
  local attempt=1
  local delay=5
  local http_code
  local response
  
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt of $max_attempts: Executing curl call to $url"
    
    if [ -n "$upload_file" ]; then
      # Upload file case
      response=$(curl -s -w "%{http_code}" --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        --request $method \
        --upload-file "$upload_file" \
        "$url" -o /tmp/curl_response.txt)
    elif [ -n "$data" ]; then
      # With data case
      response=$(curl -s -w "%{http_code}" --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        --request $method \
        --data "$data" \
        "$url" -o /tmp/curl_response.txt)
    else
      # Simple case
      response=$(curl -s -w "%{http_code}" --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
        --request $method \
        "$url" -o /tmp/curl_response.txt)
    fi
    
    http_code=$(echo "$response" | tail -n1)
    
    # Check for successful response (2xx)
    if [[ $http_code -ge 200 && $http_code -lt 300 ]]; then
      echo "Success: $method request to $url (HTTP $http_code)"
      if [ -f /tmp/curl_response.txt ]; then
        cat /tmp/curl_response.txt
        rm /tmp/curl_response.txt
      fi
      return 0
    # 404 is fine for release creation (it may already exist)
    elif [[ $method == "POST" && $http_code -eq 404 && $url == *"/releases" ]]; then
      echo "Note: Release may already exist. Continuing. (HTTP $http_code)"
      return 0
    # 409 is fine for release creation (already exists)
    elif [[ $method == "POST" && $http_code -eq 409 && $url == *"/releases" ]]; then
      echo "Note: Release already exists. Continuing. (HTTP $http_code)"
      return 0
    else
      echo "Attempt $attempt failed: $method request to $url (HTTP $http_code)"
      if [ -f /tmp/curl_response.txt ]; then
        echo "Error response:"
        cat /tmp/curl_response.txt
        rm /tmp/curl_response.txt
      fi
      
      if [ $attempt -lt $max_attempts ]; then
        echo "Retrying in $delay seconds..."
        sleep $delay
        attempt=$((attempt + 1))
        delay=$((delay * 2))
      else
        echo "Maximum attempts reached. Giving up."
        return 1
      fi
    fi
  done
  
  return 1
}

# Check for wheels
check_wheels() {
  if [ ! -d "${WHEELS_DIR}" ]; then
    echo "ERROR: Wheels directory '${WHEELS_DIR}' does not exist"
    exit 1
  fi
  
  # Count wheel files
  local wheel_count=$(find "${WHEELS_DIR}" -name "*.whl" | wc -l)
  if [ $wheel_count -eq 0 ]; then
    echo "WARNING: No wheel files found in ${WHEELS_DIR}. Nothing to upload."
    exit 0
  else
    echo "Found $wheel_count wheel files to upload."
  fi
}

# Main execution
echo "Starting GitLab release artifact upload process..."

# Validate required environment variables
check_required_vars

# Check for wheels
check_wheels

# Create a release in GitLab if it doesn't exist
echo "Creating/updating GitLab release ${TAG_NAME}..."
release_data="tag_name=${TAG_NAME}&name=${RELEASE_NAME}&description=${RELEASE_DESCRIPTION}"
if ! curl_with_retry "https://gitlab-master.nvidia.com/api/v4/projects/${GITLAB_PROJECT_ID}/releases" "POST" "$release_data"; then
  echo "WARNING: Failed to create release, but will try to upload assets anyway"
fi

# Upload each wheel file as a release artifact
upload_count=0
failed_count=0
for wheel in ${WHEELS_DIR}/*.whl; do
  if [ ! -f "$wheel" ]; then
    continue  # Skip if no matching files
  fi
  
  filename=$(basename "$wheel")
  echo "Uploading $filename as release artifact to GitLab release ${TAG_NAME}..."
  
  upload_url="https://gitlab-master.nvidia.com/api/v4/projects/${GITLAB_PROJECT_ID}/releases/${TAG_NAME}/assets/links"
  upload_data="name=${filename}&url=${filename}&link_type=package"
  
  if curl_with_retry "${upload_url}?${upload_data}" "POST" "" "$wheel"; then
    echo "Successfully uploaded $filename"
    upload_count=$((upload_count + 1))
  else
    echo "ERROR: Failed to upload $filename"
    failed_count=$((failed_count + 1))
  fi
done

# Report results
echo "Upload summary:"
echo "- Total wheels uploaded: $upload_count"
echo "- Total wheels failed: $failed_count"

if [ $failed_count -gt 0 ]; then
  echo "WARNING: Some wheels failed to upload. Check the logs for details."
  exit 1
else
  echo "All wheels have been successfully uploaded as GitLab release artifacts"
  exit 0
fi 