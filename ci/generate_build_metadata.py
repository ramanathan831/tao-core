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

"""Generate build metadata JSON file."""

import json
import sys
import os
import argparse
import subprocess
from pathlib import Path


def run_git_command(cmd, cwd=None):
    """Execute a git command and return output.
    
    Args:
        cmd: Command as list of strings
        cwd: Working directory (defaults to WORKSPACE)
    
    Returns:
        str: Command output or empty string on error
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception:
        return ""


def get_commit_info(skip=0, workspace=None):
    """Get commit information using git.
    
    Args:
        skip: Number of commits to skip
        workspace: Git repository path
    
    Returns:
        dict: Commit info or None if not available
    """
    cmd = ['git', 'log', f'-{skip+1}', f'--skip={skip}', '--pretty=format:%H|%an|%ae|%cd|%s']
    output = run_git_command(cmd, cwd=workspace)
    
    if not output:
        return None
    
    # Get the first line (should only be one commit)
    lines = output.strip().split('\n')
    if not lines or not lines[0]:
        return None
    
    parts = lines[0].split('|', 4)
    if len(parts) >= 5:
        return {
            "sha": parts[0],
            "author_name": parts[1],
            "author_email": parts[2],
            "date": parts[3],
            "message": parts[4]
        }
    return None


def check_submodules(workspace=None):
    """Check for submodules in the repository.
    
    Args:
        workspace: Git repository path
    
    Returns:
        tuple: (has_submodules: bool, status: str)
    """
    gitmodules_path = Path(workspace or '.') / '.gitmodules'
    has_submodules = gitmodules_path.exists()
    
    if has_submodules:
        status = run_git_command(['git', 'submodule', 'status'], cwd=workspace)
        if not status:
            status = "No submodules initialized"
    else:
        status = "No submodules in repository"
    
    return has_submodules, status


def generate_metadata(args):
    """Generate complete build metadata.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict: Complete metadata dictionary
    """
    workspace = args.workspace or os.environ.get('WORKSPACE', '.')
    
    # Get git information directly
    git_sha = run_git_command(['git', 'rev-parse', 'HEAD'], cwd=workspace)
    git_short_sha = run_git_command(['git', 'rev-parse', '--short', 'HEAD'], cwd=workspace)
    branch = run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=workspace)
    
    # Get commit history (top 3 commits)
    commit_history = [
        get_commit_info(0, workspace),
        get_commit_info(1, workspace),
        get_commit_info(2, workspace)
    ]
    
    # Check submodules
    has_submodules, submodule_status = check_submodules(workspace)
    
    metadata = {
        "build_info": {
            "timestamp": args.timestamp,
            "job_name": args.job_name,
            "build_number": args.build_number,
            "build_url": args.build_url,
            "build_type": args.build_type,
            "tao_version": args.tao_version,
            "wheel_version": args.wheel_version
        },
        "build_parameters": {
            "tao_version": args.tao_version,
            "rc_build": args.rc_build == 'True',
            "force_rebuild": args.force_rebuild == 'True',
            "skip_prod_promote": args.skip_prod_promote == 'True',
            "triggered_by": args.triggered_by
        },
        "build_status": {
            "result": args.build_result,
            "duration_ms": int(args.duration_ms),
            "skipped_build": args.build_skipped == 'True'
        },
        "git_info": {
            "sha": git_sha,
            "short_sha": git_short_sha,
            "branch": branch,
            "commit_history": commit_history
        },
        "artifacts": {
            "wheel_url": args.wheel_url,
            "final_wheel_url": args.final_wheel_url,
            "wheel_install_location": args.wheel_install_location,
            "final_wheel_install_location": args.final_wheel_install_location
        },
        "submodules": {
            "has_submodules": has_submodules,
            "status": submodule_status
        },
        "environment": {
            "python_version": args.python_version,
            "workspace": workspace
        }
    }
    
    # Conditionally add previous build info only if build was skipped
    if args.build_skipped == 'True' and args.previous_build_number and args.previous_build_url:
        metadata["build_status"]["previous_build_reused"] = args.previous_build_number
        metadata["build_status"]["previous_successful_build"] = args.previous_build_url
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate build metadata')
    
    # Build info (from Jenkins)
    parser.add_argument('--timestamp', required=True, help='Build timestamp')
    parser.add_argument('--job-name', required=True, help='Jenkins job name')
    parser.add_argument('--build-number', required=True, help='Jenkins build number')
    parser.add_argument('--build-url', required=True, help='Jenkins build URL')
    parser.add_argument('--build-type', required=True, help='Build type (RC/Final)')
    parser.add_argument('--tao-version', required=True, help='TAO version')
    parser.add_argument('--wheel-version', required=True, help='Wheel version')
    
    # Build parameters
    parser.add_argument('--rc-build', required=True, help='Whether this is an RC build')
    parser.add_argument('--force-rebuild', required=True, help='Whether force rebuild was requested')
    parser.add_argument('--skip-prod-promote', required=True, help='Whether to skip prod promotion')
    parser.add_argument('--triggered-by', required=True, help='User who triggered the build')
    
    # Build status (from Jenkins)
    parser.add_argument('--build-result', required=True, help='Build result (SUCCESS/FAILURE)')
    parser.add_argument('--duration-ms', required=True, help='Build duration in milliseconds')
    parser.add_argument('--build-skipped', default='False', help='Whether build was skipped (optional, defaults to False)')
    parser.add_argument('--previous-build-number', default='', help='Previous build number (optional, used if skipped)')
    parser.add_argument('--previous-build-url', default='', help='Previous build URL (optional, used if skipped)')
    
    # Artifacts
    parser.add_argument('--wheel-url', required=True, help='Wheel artifact URL')
    parser.add_argument('--final-wheel-url', required=True, help='Final wheel artifact URL')
    parser.add_argument('--wheel-install-location', required=True, help='Wheel install location')
    parser.add_argument('--final-wheel-install-location', required=True, help='Final wheel install location')
    
    # Environment
    parser.add_argument('--python-version', required=True, help='Python version')
    parser.add_argument('--workspace', default=None, help='Workspace directory (defaults to WORKSPACE env var)')
    
    # Output file
    parser.add_argument('--output', default='release_metadata.json', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        metadata = generate_metadata(args)
        
        # Write to JSON file
        with open(args.output, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n=== Release Metadata ===")
        print(json.dumps(metadata, indent=2))
        print(f"\n✓ Metadata written to {args.output}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


