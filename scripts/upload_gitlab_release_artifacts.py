#!/usr/bin/env python3
import os
import sys
import glob
import re
import time
import requests
import tempfile
from pathlib import Path

def check_required_vars():
    """Check if required environment variables are set."""
    missing_vars = 0
    
    if not os.environ.get("GITLAB_TOKEN"):
        print("ERROR: GITLAB_TOKEN environment variable is not set")
        missing_vars = 1
    
    if not os.environ.get("GITLAB_PROJECT_ID"):
        print("ERROR: GITLAB_PROJECT_ID environment variable is not set")
        missing_vars = 1
    
    if not os.environ.get("TAG_NAME"):
        print("ERROR: TAG_NAME environment variable is not set")
        missing_vars = 1
    
    if not os.environ.get("WHEELS_DIR") and not os.environ.get("APT_DIR"):
        print("WARNING: Neither WHEELS_DIR nor APT_DIR environment variable is set, defaulting to 'third_party_wheels' and 'third_party_apt'")
        os.environ["WHEELS_DIR"] = "third_party_wheels"
        os.environ["APT_DIR"] = "third_party_apt"
    elif not os.environ.get("WHEELS_DIR"):
        print("WARNING: WHEELS_DIR environment variable is not set, defaulting to 'third_party_wheels'")
        os.environ["WHEELS_DIR"] = "third_party_wheels"
    elif not os.environ.get("APT_DIR"):
        print("WARNING: APT_DIR environment variable is not set, defaulting to 'third_party_apt'")
        os.environ["APT_DIR"] = "third_party_apt"
    
    if not os.environ.get("RELEASE_NAME"):
        print(f"WARNING: RELEASE_NAME environment variable is not set, defaulting to 'Release {os.environ.get('TAG_NAME')}'")
        os.environ["RELEASE_NAME"] = f"Release {os.environ.get('TAG_NAME')}"
    
    if not os.environ.get("RELEASE_DESCRIPTION"):
        print(f"WARNING: RELEASE_DESCRIPTION environment variable is not set, defaulting to 'Third-party packages for {os.environ.get('TAG_NAME')}'")
        os.environ["RELEASE_DESCRIPTION"] = f"Third-party packages for {os.environ.get('TAG_NAME')}"
    
    if missing_vars == 1:
        print("ERROR: Required environment variables are missing. Exiting.")
        sys.exit(1)

def curl_with_retry(url, method="GET", data=None, upload_file=None, max_attempts=3):
    """Execute a request with retries."""
    attempt = 1
    delay = 5
    
    headers = {"PRIVATE-TOKEN": os.environ.get("GITLAB_TOKEN")}
    
    while attempt <= max_attempts:
        print(f"Attempt {attempt} of {max_attempts}: Executing request to {url}")
        
        try:
            if upload_file:
                # Upload file case
                with open(upload_file, 'rb') as f:
                    response = requests.request(method, url, headers=headers, files={'file': f}, timeout=60)
            elif data:
                # With data case
                response = requests.request(method, url, headers=headers, data=data, timeout=60)
            else:
                # Simple case
                response = requests.request(method, url, headers=headers, timeout=60)
            
            http_code = response.status_code
            
            # Check for successful response (2xx)
            if 200 <= http_code < 300:
                print(f"Success: {method} request to {url} (HTTP {http_code})")
                if response.text:
                    print(response.text)
                return True
            # 404 is fine for release creation (it may already exist)
            elif method == "POST" and http_code == 404 and "/releases" in url:
                print(f"Note: Release may already exist. Continuing. (HTTP {http_code})")
                return True
            # 409 is fine for release creation (already exists)
            elif method == "POST" and http_code == 409 and "/releases" in url:
                print(f"Note: Release already exists. Continuing. (HTTP {http_code})")
                return True
            # Package already exists is often 400 or 409
            elif ("/packages/pypi" in url or "/packages/generic" in url) and (http_code == 400 or http_code == 409):
                print(f"Note: Package might already exist. Continuing. (HTTP {http_code})")
                if response.text:
                    print(f"Response: {response.text}")
                return True
            else:
                print(f"Attempt {attempt} failed: {method} request to {url} (HTTP {http_code})")
                if response.text:
                    print(f"Error response: {response.text}")
                
                if attempt < max_attempts:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempt += 1
                    delay *= 2
                else:
                    print("Maximum attempts reached. Giving up.")
                    return False
                
        except Exception as e:
            print(f"Exception during {method} request to {url}: {str(e)}")
            if attempt < max_attempts:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
                delay *= 2
            else:
                print("Maximum attempts reached. Giving up.")
                return False
    
    return False

def check_packages(directory, file_pattern):
    """
    Check for packages in the specified directory.
    
    :param directory: Directory to check for packages
    :param file_pattern: File pattern to match (e.g., "*.whl", "*.deb")
    :return: List of matching files
    """
    if not os.path.isdir(directory):
        print(f"WARNING: Directory '{directory}' does not exist")
        return []
    
    # Count files
    files = glob.glob(f"{directory}/{file_pattern}")
    file_count = len(files)
    
    if file_count == 0:
        print(f"WARNING: No {file_pattern} files found in {directory}.")
    else:
        print(f"Found {file_count} {file_pattern} files in {directory} to upload.")
    
    return files

def extract_package_info(file_path):
    """
    Extract package information from filename.
    
    :param file_path: Path to the package file
    :return: Tuple of (package_name, version)
    """
    filename = os.path.basename(file_path)
    
    # Check if it's a wheel file
    if filename.endswith('.whl'):
        # Parse wheel filename: {dist}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
        parts = filename.split('-')
        if len(parts) >= 4:
            package_name = parts[0]
            version = parts[1]
            return package_name, version
    
    # Check if it's a deb file
    elif filename.endswith('.deb'):
        # Parse deb filename: {package}_{version}_{architecture}.deb
        parts = filename.split('_')
        if len(parts) >= 2:
            package_name = parts[0]
            version = parts[1].split('_')[0]  # Get version part
            return package_name, version
    
    # For other file types or if parsing fails
    package_name = os.path.splitext(filename)[0]
    return package_name, "unknown"

def create_release():
    """Create a release in GitLab."""
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    tag_name = os.environ.get("TAG_NAME")
    release_name = os.environ.get("RELEASE_NAME")
    release_description = os.environ.get("RELEASE_DESCRIPTION")
    
    print(f"Creating/updating GitLab release {tag_name}...")
    release_data = {
        "tag_name": tag_name,
        "name": release_name,
        "description": release_description
    }
    
    release_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases"
    if not curl_with_retry(release_url, "POST", release_data):
        print("WARNING: Failed to create release, but will try to upload packages anyway")

def upload_pypi_package(file_path, gitlab_project_id):
    """
    Upload a Python wheel to GitLab PyPI package registry.
    
    :param file_path: Path to the wheel file
    :param gitlab_project_id: GitLab project ID
    :return: True if successful, False otherwise
    """
    print(f"Uploading PyPI package: {os.path.basename(file_path)}")
    
    # Upload to PyPI package registry
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/packages/pypi"
    return curl_with_retry(upload_url, "POST", None, file_path)

def upload_generic_package(file_path, gitlab_project_id, package_type="apt"):
    """
    Upload a generic package (e.g., .deb) to GitLab generic package registry.
    
    :param file_path: Path to the package file
    :param gitlab_project_id: GitLab project ID
    :param package_type: Package type identifier (e.g., "apt", "rpm")
    :return: True if successful, False otherwise
    """
    filename = os.path.basename(file_path)
    package_name, version = extract_package_info(file_path)
    
    print(f"Uploading generic {package_type} package: {filename}")
    
    # Upload to generic package registry
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/packages/generic/{package_type}/{package_name}/{version}/{filename}"
    return curl_with_retry(upload_url, "PUT", None, file_path)

def create_package_link(filename, package_name, gitlab_project_id, tag_name):
    """
    Create a link in the release to a package.
    
    :param filename: Name of the package file
    :param package_name: Name of the package
    :param gitlab_project_id: GitLab project ID
    :param tag_name: Release tag name
    :return: True if successful, False otherwise
    """
    print(f"Creating link in release {tag_name} to package {package_name}...")
    
    # Link directly to the GitLab package page
    package_url = f"https://gitlab-master.nvidia.com/projects/{gitlab_project_id}/packages/{package_name}"
    link_data = {
        "name": filename,
        "url": package_url,
        "link_type": "package"
    }
    
    return curl_with_retry(
        f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases/{tag_name}/assets/links", 
        "POST", 
        link_data
    )

def main():
    """Main function to upload GitLab release artifacts."""
    print("Starting GitLab package and release artifact upload process...")
    
    # Validate required environment variables
    check_required_vars()
    
    # Get environment variables
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    tag_name = os.environ.get("TAG_NAME")
    wheels_dir = os.environ.get("WHEELS_DIR")
    apt_dir = os.environ.get("APT_DIR")
    
    # Check for packages
    wheel_files = check_packages(wheels_dir, "*.whl")
    apt_files = check_packages(apt_dir, "*.deb")
    
    if not wheel_files and not apt_files:
        print("No packages found to upload. Exiting.")
        return 1
    
    # Create a release in GitLab if it doesn't exist
    create_release()
    
    # Upload each wheel file to GitLab package registry and link it in the release
    upload_count = 0
    failed_count = 0
    
    # Upload Python wheels
    for wheel in wheel_files:
        filename = os.path.basename(wheel)
        package_name, version = extract_package_info(wheel)
        
        print(f"Processing wheel: {filename} ({package_name} v{version})")
        
        if upload_pypi_package(wheel, gitlab_project_id):
            print(f"Successfully uploaded {filename} to package registry")
            
            if create_package_link(filename, package_name, gitlab_project_id, tag_name):
                print(f"Successfully linked {filename} in release")
                upload_count += 1
            else:
                print(f"WARNING: Uploaded package but failed to link it in the release")
                failed_count += 1
        else:
            print(f"ERROR: Failed to upload {filename} to package registry")
            failed_count += 1
    
    # Upload APT packages
    for apt_file in apt_files:
        filename = os.path.basename(apt_file)
        package_name, version = extract_package_info(apt_file)
        
        print(f"Processing APT package: {filename} ({package_name} v{version})")
        
        if upload_generic_package(apt_file, gitlab_project_id, package_type="apt"):
            print(f"Successfully uploaded {filename} to package registry")
            
            if create_package_link(filename, f"apt:{package_name}", gitlab_project_id, tag_name):
                print(f"Successfully linked {filename} in release")
                upload_count += 1
            else:
                print(f"WARNING: Uploaded package but failed to link it in the release")
                failed_count += 1
        else:
            print(f"ERROR: Failed to upload {filename} to package registry")
            failed_count += 1
    
    # Report results
    print("Upload summary:")
    print(f"- Total packages uploaded and linked: {upload_count}")
    print(f"- Total packages failed: {failed_count}")
    
    if failed_count > 0:
        print("WARNING: Some packages failed to upload or link. Check the logs for details.")
        return 1
    else:
        print("All packages have been successfully uploaded to GitLab package registry and linked in the release")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 