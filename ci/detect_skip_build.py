#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect if build should be skipped by downloading and comparing previous build metadata."""

import json
import sys
import argparse
import subprocess
import os
from pathlib import Path


def download_metadata(artifact_url, output_file='previous_release_metadata.json'):
    """Download metadata artifact from Jenkins using authenticated curl.
    
    Credentials are read from JENKINS_USER and JENKINS_TOKEN environment variables
    to avoid exposing them in process lists.
    
    Args:
        artifact_url: Full URL to the artifact
        output_file: Where to save the downloaded file
        
    Returns:
        bool: True if download and validation succeeded
    """
    try:
        # Read credentials from environment (already validated by caller)
        username = os.environ['JENKINS_USER']
        password = os.environ['JENKINS_TOKEN']
        
        # Use curl with config file via stdin to avoid credential exposure
        curl_config = f"user = \"{username}:{password}\""
        
        cmd = [
            'curl',
            '--config', '-',  # Read config from stdin
            '-f',  # Fail silently on HTTP errors
            '-s',  # Silent mode
            '-o', output_file,
            artifact_url
        ]
        
        result = subprocess.run(cmd, input=curl_config, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"⚠ Failed to download metadata (curl exit code: {result.returncode})", file=sys.stderr)
            return False
        
        # Verify file exists and is non-empty
        if not Path(output_file).exists() or Path(output_file).stat().st_size == 0:
            print(f"⚠ Downloaded file is missing or empty", file=sys.stderr)
            return False
        
        # Validate JSON
        try:
            with open(output_file, 'r') as f:
                json.load(f)
            print(f"✓ Retrieved and validated metadata from {artifact_url}")
            return True
        except json.JSONDecodeError as e:
            print(f"⚠ Downloaded file is not valid JSON: {e}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"ERROR downloading metadata: {str(e)}", file=sys.stderr)
        return False


def compare_metadata(metadata_file, current_sha, current_version, current_build_type):
    """Compare previous build metadata with current build parameters.
    
    Args:
        metadata_file: Path to previous metadata JSON file
        current_sha: Current git SHA
        current_version: Current TAO version
        current_build_type: Current build type (RC/Final)
        
    Returns:
        bool: True if builds are identical and current build can be skipped
    """
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        last_sha = metadata.get('git_info', {}).get('sha', '')
        last_version = metadata.get('build_info', {}).get('tao_version', '')
        last_build_type = metadata.get('build_info', {}).get('build_type', '')
        
        print(f"Comparing builds:")
        print(f"  Current SHA: {current_sha}")
        print(f"  Last SHA:    {last_sha}")
        print(f"  Current Version: {current_version}")
        print(f"  Last Version:    {last_version}")
        print(f"  Current Build Type: {current_build_type}")
        print(f"  Last Build Type:    {last_build_type}")
        
        # All three must match for builds to be considered identical
        should_skip = (
            current_sha == last_sha and
            current_version == last_version and
            current_build_type == last_build_type
        )
        
        return should_skip
        
    except Exception as e:
        print(f"ERROR comparing metadata: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download previous build metadata and determine if current build should be skipped'
    )
    parser.add_argument('--artifact-url', required=True, help='URL to previous build metadata artifact')
    parser.add_argument('--current-sha', required=True, help='Current git SHA')
    parser.add_argument('--current-version', required=True, help='Current TAO version')
    parser.add_argument('--current-build-type', required=True, help='Current build type (RC/Final)')
    
    args = parser.parse_args()
    
    # Get Jenkins credentials from environment
    jenkins_user = os.environ.get('JENKINS_USER')
    jenkins_token = os.environ.get('JENKINS_TOKEN')
    
    if not jenkins_user or not jenkins_token:
        print("ERROR: JENKINS_USER and JENKINS_TOKEN environment variables must be set", file=sys.stderr)
        sys.exit(1)
    
    # Download and validate metadata
    print(f"Downloading metadata from: {args.artifact_url}")
    output_file = 'previous_release_metadata.json'
    download_success = download_metadata(
        args.artifact_url,
        output_file
    )
    
    if not download_success:
        print("⚠ No metadata found in last successful build - proceeding with full build")
        print("SHOULD_SKIP=false")
        sys.exit(0)
    
    # Compare metadata
    should_skip = compare_metadata(
        output_file,
        args.current_sha,
        args.current_version,
        args.current_build_type
    )
    
    # Output result
    if should_skip:
        print("\n=== ✓ Builds are identical - can skip ===")
    else:
        print("\n=== ✗ Changes detected - must build ===")
    
    print(f"SHOULD_SKIP={'true' if should_skip else 'false'}")


if __name__ == '__main__':
    main()


