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

# Create a release in GitLab if it doesn't exist
curl --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --data "tag_name=${TAG_NAME}" \
  --data "name=${RELEASE_NAME}" \
  --data "description=${RELEASE_DESCRIPTION}" \
  "https://gitlab-master.nvidia.com/api/v4/projects/${GITLAB_PROJECT_ID}/releases" || true

# Upload each wheel file as a release artifact
for wheel in ${WHEELS_DIR}/*.whl; do
  filename=$(basename $wheel)
  echo "Uploading $filename as release artifact to GitLab release ${TAG_NAME}..."
  curl --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --upload-file "$wheel" \
    "https://gitlab-master.nvidia.com/api/v4/projects/${GITLAB_PROJECT_ID}/releases/${TAG_NAME}/assets/links?name=${filename}&url=${filename}&link_type=package"
done

echo "All wheels have been uploaded as GitLab release artifacts" 