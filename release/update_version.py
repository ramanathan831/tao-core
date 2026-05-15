#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TAO PyTorch build-time version patcher.
This script patches version numbers in Dockerfile and version.py during Jenkins builds.
Version is passed from Jenkins pipeline, keeping version management in Jenkinsfiles.
"""

import os
import re
import argparse
import sys
from pathlib import Path

def update_version_py(major, minor, patch, root_dir):
    """Update release/python/version.py with new version components."""
    filepath = os.path.join(root_dir, "release", "python", "version.py")
    
    if not os.path.exists(filepath):
        print(f"⚠ Warning: {filepath} not found")
        return
    
    print(f"Updating release/python/version.py...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update MAJOR, MINOR, PATCH
    content = re.sub(r'MAJOR = "[^"]*"', f'MAJOR = "{major}"', content)
    content = re.sub(r'MINOR = "[^"]*"', f'MINOR = "{minor}"', content)
    content = re.sub(r'PATCH = "[^"]*"', f'PATCH = "{patch}"', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Updated release/python/version.py")

def update_dockerfile(version, root_dir):
    """Update TAO_TOOLKIT_VERSION in Dockerfile (optional - skipped if not found)."""
    filepath = os.path.join(root_dir, "release", "docker", "Dockerfile")
    
    if not os.path.exists(filepath):
        print(f"ℹ Skipping Dockerfile update (not found at {filepath})")
        return False
    
    print(f"Updating release/docker/Dockerfile...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update TAO_TOOLKIT_VERSION
    pattern = r'ENV TAO_TOOLKIT_VERSION="[^"]*"'
    replacement = f'ENV TAO_TOOLKIT_VERSION="{version}"'
    content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Updated release/docker/Dockerfile")
    return True

def verify_updates(version, root_dir):
    """Verify that build files have been updated correctly."""
    print(f"\n🔍 Verifying version updates to {version}...")
    
    issues = []
    
    # Check version.py
    version_py_path = os.path.join(root_dir, "release", "python", "version.py")
    if os.path.exists(version_py_path):
        with open(version_py_path, 'r') as f:
            content = f.read()
        major, minor, patch = version.split('.')
        if (f'MAJOR = "{major}"' not in content or 
            f'MINOR = "{minor}"' not in content or 
            f'PATCH = "{patch}"' not in content):
            issues.append("release/python/version.py: Version components not updated correctly")
    
    # Check Dockerfile (optional)
    dockerfile_path = os.path.join(root_dir, "release", "docker", "Dockerfile")
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        if f'ENV TAO_TOOLKIT_VERSION="{version}"' not in content:
            issues.append("release/docker/Dockerfile: TAO_TOOLKIT_VERSION not updated correctly")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Build files updated successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Patch TAO PyTorch build files with version from Jenkins")
    parser.add_argument("--version", required=True, help="Version to set (e.g., 6.25.7). Required parameter from Jenkins.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify current version consistency")
    parser.add_argument("--root-dir", default=".", help="Root directory of the project (default: current directory)")
    
    args = parser.parse_args()
    
    # Resolve root directory to absolute path
    root_dir = os.path.abspath(args.root_dir)
    
    # Parse provided version
    try:
        major, minor, patch = args.version.split('.')
        version = args.version
    except ValueError:
        print("❌ Error: Version must be in format MAJOR.MINOR.PATCH (e.g., 6.25.7)")
        sys.exit(1)
    
    print(f"🚀 TAO PyTorch Build-Time Version Patcher")
    print(f"📁 Working directory: {root_dir}")
    print(f"🏷️  Target version: {version}")
    
    if args.verify_only:
        success = verify_updates(version, root_dir)
        sys.exit(0 if success else 1)
    
    # Update build files only
    print(f"\n📝 Patching build files with version {version}...")
    
    try:
        update_version_py(major, minor, patch, root_dir)
        dockerfile_updated = update_dockerfile(version, root_dir)
        
        # Verify updates
        success = verify_updates(version, root_dir)
        
        if success:
            files_updated = ["release/python/version.py"]
            if dockerfile_updated:
                files_updated.append("release/docker/Dockerfile")
            print(f"\n🎉 Successfully patched build files to version {version}")
            print(f"📝 Updated files: {', '.join(files_updated)}")
            sys.exit(0)
        else:
            print(f"\n❌ Some files were not updated correctly")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()