#!/usr/bin/env python3
import os
import sys
import glob
import time
import requests

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
    
    if not os.environ.get("APT_DIR"):
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

def test_gitlab_connectivity():
    """Test GitLab API connectivity and token validity."""
    print("Testing GitLab API connectivity...")
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    gitlab_token = os.environ.get("GITLAB_TOKEN")
    
    # First check if we can access the GitLab API
    url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}"
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"Successfully connected to GitLab API. Project name: {response.json().get('name', 'Unknown')}")
            return True
        else:
            print(f"Failed to connect to GitLab API. Status code: {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            
            # Check different permission issues
            if response.status_code == 404:
                print("ERROR: Project not found or token does not have access to this project")
            elif response.status_code == 401:
                print("ERROR: Authentication failed. Check your GitLab token")
            elif response.status_code == 403:
                print("ERROR: Permission denied. Token does not have sufficient permissions")
            
            # Test general API access
            try:
                user_response = requests.get("https://gitlab-master.nvidia.com/api/v4/user", headers=headers, timeout=10)
                if user_response.status_code == 200:
                    print(f"Token has API access (user: {user_response.json().get('username', 'Unknown')}), but not for this project")
                else:
                    print("Token does not have basic API access")
            except Exception as e:
                print(f"Error testing user access: {str(e)}")
            
            return False
    except Exception as e:
        print(f"Exception testing GitLab API connectivity: {str(e)}")
        return False

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
                return response.json() if response.text and response.headers.get('Content-Type', '').startswith('application/json') else True
            # 404 is fine for release creation (it may already exist)
            elif method == "POST" and http_code == 404 and "/releases" in url:
                print(f"Note: Release may already exist. Continuing. (HTTP {http_code})")
                return True
            # 409 is fine for release creation (already exists)
            elif method == "POST" and http_code == 409 and "/releases" in url:
                print(f"Note: Release already exists. Continuing. (HTTP {http_code})")
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

def check_apt_packages(directory):
    """
    Check for APT packages in the specified directory.
    
    :param directory: Directory to check for APT packages
    :return: List of matching deb files
    """
    if not os.path.isdir(directory):
        print(f"WARNING: Directory '{directory}' does not exist")
        return []
    
    # Count files
    files = glob.glob(f"{directory}/*.deb")
    file_count = len(files)
    
    if file_count == 0:
        print(f"WARNING: No .deb files found in {directory}.")
    else:
        print(f"Found {file_count} .deb files in {directory} to upload.")
    
    return files

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
        print("WARNING: Failed to create release, but will try to upload artifacts anyway")

def upload_file_to_gitlab(file_path, gitlab_project_id):
    """
    Upload file to GitLab project's uploads storage.
    This is a first step in adding a file as a release asset.
    
    :param file_path: Path to the file to upload
    :param gitlab_project_id: GitLab project ID
    :return: Response JSON containing upload info, or False if failed
    """
    filename = os.path.basename(file_path)
    print(f"Uploading file to GitLab project storage: {filename}")
    
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/uploads"
    response = curl_with_retry(upload_url, "POST", None, file_path)
    
    if response and isinstance(response, dict):
        print(f"Successfully uploaded file to project storage: {filename}")
        return response
    else:
        print(f"ERROR: Failed to upload file to project storage: {filename}")
        return False

def add_file_as_release_asset(upload_info, gitlab_project_id, tag_name):
    """
    Add an uploaded file as a release asset.
    
    :param upload_info: Upload info from GitLab uploads API
    :param gitlab_project_id: GitLab project ID
    :param tag_name: Release tag name
    :return: True if successful, False otherwise
    """
    if not upload_info or not isinstance(upload_info, dict):
        return False

    asset_url = f"https://gitlab-master.nvidia.com{upload_info.get('url', '')}"
    asset_name = upload_info.get('alt', os.path.basename(upload_info.get('url', '')))
    
    print(f"Adding file as release asset: {asset_name}")
    
    link_data = {
        "name": asset_name,
        "url": asset_url,
        "link_type": "other"  # Use "other" for direct downloads
    }
    
    assets_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases/{tag_name}/assets/links"
    if curl_with_retry(assets_url, "POST", link_data):
        print(f"Successfully added file as release asset: {asset_name}")
        return True
    else:
        print(f"ERROR: Failed to add file as release asset: {asset_name}")
        return False

def main():
    """Main function to upload APT packages to GitLab release as artifacts."""
    print("Starting GitLab APT package upload process (direct artifact mode)...")
    
    # Print script version for debugging
    print("Script version: 2.0.0 (APT direct artifacts)")
    
    # Print environment variables for debugging if DEBUG is set
    if os.environ.get("DEBUG"):
        print("DEBUG: Environment variables:")
        for key in ["GITLAB_PROJECT_ID", "TAG_NAME", "RELEASE_NAME", "APT_DIR"]:
            print(f"DEBUG: {key}={os.environ.get(key, 'Not set')}")
    
    # Validate required environment variables
    check_required_vars()
    
    # Test GitLab API connectivity first
    if not test_gitlab_connectivity():
        print("ERROR: Failed to connect to GitLab API. Exiting.")
        return 1
    
    # Get environment variables
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    tag_name = os.environ.get("TAG_NAME")
    apt_dir = os.environ.get("APT_DIR")
    
    # Check for APT packages
    apt_files = check_apt_packages(apt_dir)
    
    if not apt_files:
        print("No APT packages found to upload. Exiting.")
        return 1
    
    # Create a release in GitLab if it doesn't exist
    create_release()
    
    # Upload each APT file to GitLab and add as release asset
    upload_count = 0
    failed_count = 0
    
    for apt_file in apt_files:
        filename = os.path.basename(apt_file)
        print(f"Processing APT package: {filename}")
        
        # Upload file to GitLab first
        upload_info = upload_file_to_gitlab(apt_file, gitlab_project_id)
        
        if upload_info:
            # Add uploaded file as release asset
            if add_file_as_release_asset(upload_info, gitlab_project_id, tag_name):
                upload_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1
    
    # Report results
    print("Upload summary:")
    print(f"- Total APT artifacts uploaded and linked: {upload_count}")
    print(f"- Total APT artifacts failed: {failed_count}")
    
    if failed_count > 0:
        print("WARNING: Some APT packages failed to upload or link. Check the logs for details.")
        return 1
    else:
        print("All APT packages have been successfully uploaded as release artifacts")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 