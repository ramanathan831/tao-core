#!/bin/bash
set -e

# Create output directory for third party wheels
mkdir -p third_party_wheels

echo "Starting download of third-party wheels..."

# Function to attempt download with retries
download_with_retry() {
  local max_attempts=3
  local attempt=1
  local delay=5
  
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt of $max_attempts: Downloading binary wheels..."
    if pip download --only-binary=:all: -r requirements.txt -d third_party_wheels --no-deps; then
      echo "Successfully downloaded all binary wheels!"
      return 0
    else
      echo "Attempt $attempt failed. Retrying in $delay seconds..."
      sleep $delay
      attempt=$((attempt + 1))
      delay=$((delay * 2))
    fi
  done
  
  echo "Could not download all wheels as binaries after $max_attempts attempts."
  return 1
}

# Try to download binary wheels first (preferred)
if ! download_with_retry; then
  echo "Falling back to downloading packages individually..."
  
  # Read requirements.txt line by line
  while IFS= read -r package || [[ -n "$package" ]]; do
    # Skip comments and empty lines
    if [[ $package == \#* ]] || [[ -z "${package// }" ]]; then
      continue
    fi
    
    echo "Processing package: $package"
    
    # Try multiple methods for each package
    if pip download --only-binary=:all: "$package" -d third_party_wheels --no-deps; then
      echo "Successfully downloaded binary wheel for $package"
    elif pip download --only-binary=:all: "$package" -d third_party_wheels --no-deps --extra-index-url https://pypi.org/simple; then
      echo "Successfully downloaded binary wheel for $package from PyPI"
    elif pip download --no-binary=:none: "$package" -d third_party_wheels --no-deps; then
      echo "Warning: Could only download source distribution for $package"
      # If only got a source distribution, try to build a wheel
      for src in third_party_wheels/*"$package"*.tar.gz; do
        if [ -f "$src" ]; then
          echo "Attempting to build wheel from source for $package..."
          pip wheel "$src" -w third_party_wheels --no-deps || true
          # Clean up source distribution if successfully built a wheel
          if ls third_party_wheels/*"$package"*.whl 1> /dev/null 2>&1; then
            rm -f "$src"
          fi
        fi
      done
    else
      echo "Warning: Failed to download $package, continuing with other packages"
    fi
  done < requirements.txt
fi

# Check what we've downloaded
echo "Summary of downloaded wheels:"
ls -la third_party_wheels/

# Always return success
echo "Third-party wheel download process completed."
exit 0 