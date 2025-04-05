#!/usr/bin/env python3
import os
import sys
import glob
import time
import json
import requests
import traceback

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
    
    url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}"
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    try:
        print(f"DEBUG: Testing connectivity to {url}")
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"Successfully connected to GitLab API. Project name: {response.json().get('name', 'Unknown')}")
            return True
        else:
            print(f"Failed to connect to GitLab API. Status code: {response.status_code}")
            if response.text:
                print(f"Error response: {response.text}")
            
            if response.status_code == 404:
                print("ERROR: Project not found or token does not have access to this project")
            elif response.status_code == 401:
                print("ERROR: Authentication failed. Check your GitLab token")
            elif response.status_code == 403:
                print("ERROR: Permission denied. Token does not have sufficient permissions")
            
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
        traceback.print_exc()
        return False

def check_gitlab_api_permissions():
    """Check if token has necessary permissions for uploads and releases."""
    print("Checking GitLab API token permissions...")
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    gitlab_token = os.environ.get("GITLAB_TOKEN")
    
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    # Check permissions needed for this script
    permissions = {
        "read_project": False,
        "upload_files": False,
        "manage_releases": False
    }
    
    # 1. Check basic project access (already tested in test_gitlab_connectivity but including here for completeness)
    project_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}"
    try:
        response = requests.get(project_url, headers=headers, timeout=30)
        if response.status_code == 200:
            permissions["read_project"] = True
            print("✓ Token has read access to the project")
        else:
            print(f"✗ Token lacks read access to the project (HTTP {response.status_code})")
    except Exception as e:
        print(f"Error checking project access: {str(e)}")
    
    # 2. Check ability to upload files (test with a small dummy upload if possible)
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/uploads"
    try:
        # Create a small temporary file for testing
        temp_file = "temp_test_upload.txt"
        with open(temp_file, "w") as f:
            f.write("Test upload permission")
        
        with open(temp_file, "rb") as f:
            files = {"file": ("test_permission.txt", f)}
            response = requests.post(upload_url, headers=headers, files=files, timeout=30)
            
            if response.status_code >= 200 and response.status_code < 300:
                permissions["upload_files"] = True
                print("✓ Token has permission to upload files")
            else:
                print(f"✗ Token lacks permission to upload files (HTTP {response.status_code})")
                if response.text:
                    print(f"  Error: {response.text[:200]}")
        
        # Clean up the temporary file
        try:
            os.remove(temp_file)
        except:
            pass
    except Exception as e:
        print(f"Error checking upload permission: {str(e)}")
        traceback.print_exc()
    
    # 3. Check releases API access
    releases_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases"
    try:
        response = requests.get(releases_url, headers=headers, timeout=30)
        if response.status_code >= 200 and response.status_code < 300:
            permissions["manage_releases"] = True
            print("✓ Token has permission to manage releases")
        else:
            print(f"✗ Token lacks permission to access releases (HTTP {response.status_code})")
            if response.text:
                print(f"  Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error checking releases permission: {str(e)}")
    
    # Summarize findings
    print("\nPermission summary:")
    all_permissions_granted = all(permissions.values())
    for perm, granted in permissions.items():
        status = "✓" if granted else "✗"
        print(f"  {status} {perm}")
    
    if all_permissions_granted:
        print("\nToken has all required permissions for this script.")
        return True
    else:
        print("\nWARNING: Token is missing some permissions required for this script.")
        print("This may cause failures during execution.")
        return False

def get_project_path(gitlab_project_id, gitlab_token):
    """Fetch the project's path_with_namespace from GitLab API."""
    print(f"Fetching project path for project ID: {gitlab_project_id}")
    url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}"
    headers = {"PRIVATE-TOKEN": gitlab_token}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            project_data = response.json()
            path = project_data.get('path_with_namespace')
            if path:
                print(f"Successfully fetched project path: {path}")
                return path
            else:
                print("ERROR: 'path_with_namespace' not found in project data.")
                return None
        else:
            print(f"ERROR: Failed to fetch project data (HTTP {response.status_code})")
            if response.text:
                print(f"Error response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception fetching project path: {str(e)}")
        traceback.print_exc()
        return None

def curl_with_retry(url, method="GET", data=None, upload_file=None, max_attempts=3):
    """Execute a request with retries."""
    attempt = 1
    delay = 5
    
    headers = {"PRIVATE-TOKEN": os.environ.get("GITLAB_TOKEN")}
    current_file = upload_file if upload_file else "N/A"
    
    while attempt <= max_attempts:
        print(f"Attempt {attempt} of {max_attempts}: Executing {method} request to {url} (File: {os.path.basename(current_file)})")
        
        try:
            if upload_file:
                # Upload file case
                print(f"DEBUG: Uploading file {upload_file} to {url}")
                with open(upload_file, 'rb') as f:
                    files = {'file': (os.path.basename(upload_file), f)}
                    response = requests.request(method, url, headers=headers, files=files, timeout=180)
            elif data:
                # With data case
                if isinstance(data, dict):
                    headers["Content-Type"] = "application/json"
                    print(f"DEBUG: Sending JSON data to {url}: {data}")
                    response = requests.request(method, url, headers=headers, json=data, timeout=60)
                else:
                    print(f"DEBUG: Sending data to {url}: {data}")
                    response = requests.request(method, url, headers=headers, data=data, timeout=60)
            else:
                # Simple case
                response = requests.request(method, url, headers=headers, timeout=60)
            
            http_code = response.status_code
            
            # Print response headers for debugging
            print(f"DEBUG: Response headers for {url}: {dict(response.headers)}")
            
            # Check for successful response (2xx)
            if 200 <= http_code < 300:
                print(f"Success: {method} request to {url} (HTTP {http_code})")
                if response.text:
                    print(f"DEBUG: Response body (first 500 chars): {response.text[:500]}")
                try:
                    if response.text and 'application/json' in response.headers.get('Content-Type', ''):
                        return response.json()
                    return True
                except ValueError as e:
                    print(f"WARNING: Failed to parse JSON response from {url} despite success code: {e}")
                    print(f"DEBUG: Full response text: {response.text}")
                    return True
            # 404 is fine for release creation (it may already exist)
            elif method == "POST" and http_code == 404 and "/releases" in url:
                print(f"Note: Release may already exist. Continuing. (HTTP {http_code})")
                return True
            # 409 is fine for release creation (already exists)
            elif method == "POST" and http_code == 409 and "/releases" in url:
                print(f"Note: Release already exists. Continuing. (HTTP {http_code})")
                return True
            else:
                print(f"Attempt {attempt} failed: {method} request to {url} (HTTP {http_code}) (File: {os.path.basename(current_file)})")
                if response.text:
                    print(f"Error response body: {response.text}")
                
                if attempt < max_attempts:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempt += 1
                    delay *= 2
                else:
                    print(f"Maximum attempts reached for {url}. Giving up.")
                    return False
                
        except requests.exceptions.RequestException as e:
            print(f"Network/Request Exception during {method} request to {url} (File: {os.path.basename(current_file)}): {str(e)}")
            traceback.print_exc()
            if attempt < max_attempts:
                print(f"Retrying in {delay} seconds due to network error...")
                time.sleep(delay)
                attempt += 1
                delay *= 2
            else:
                print("Maximum attempts reached after network error. Giving up.")
                return False
        except Exception as e:
            print(f"Unexpected Exception during {method} request to {url} (File: {os.path.basename(current_file)}): {str(e)}")
            traceback.print_exc()
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
        for file in files:
            try:
                print(f"  - {os.path.basename(file)} ({os.path.getsize(file)} bytes)")
            except OSError as e:
                print(f"  - {os.path.basename(file)} (Error getting size: {e})")
    
    return files

def create_release():
    """Create a release in GitLab if it doesn't exist."""
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
    tag_name = os.environ.get("TAG_NAME")
    release_name = os.environ.get("RELEASE_NAME")
    release_description = os.environ.get("RELEASE_DESCRIPTION")
    gitlab_token = os.environ.get("GITLAB_TOKEN")
    
    if not all([gitlab_project_id, tag_name, release_name, release_description, gitlab_token]):
        print("ERROR: Missing environment variables for release creation.")
        return False

    # Check if the release already exists
    print(f"DEBUG: Checking if release {tag_name} already exists")
    check_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases/{tag_name}"
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    try:
        check_response = requests.get(check_url, headers=headers, timeout=30)
        if check_response.status_code == 200:
            print(f"Release {tag_name} already exists, skipping creation")
            return True
        elif check_response.status_code == 404:
            print(f"Release {tag_name} does not exist, proceeding with creation")
        else:
            print(f"Warning: Unexpected status code {check_response.status_code} when checking for release {tag_name}")
            if check_response.text:
                print(f"Response: {check_response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error checking for existing release {tag_name}: {e}")
        print("Proceeding with creation attempt...")

    # Check if the tag exists in the repository
    tag_check_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/repository/tags/{tag_name}"
    try:
        tag_response = requests.get(tag_check_url, headers=headers, timeout=30)
        if tag_response.status_code != 200:
            print(f"Warning: Tag {tag_name} does not exist in the repository. Creating it first.")
            # Try to create the tag if it doesn't exist
            # This requires a commit SHA to base the tag on
            # Get default branch as reference
            project_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}"
            project_response = requests.get(project_url, headers=headers, timeout=30)
            if project_response.status_code == 200:
                default_branch = project_response.json().get('default_branch', 'main')
                print(f"Using default branch '{default_branch}' as reference for creating tag")
            else:
                default_branch = 'main'
                print(f"Could not determine default branch, using '{default_branch}' as fallback")
                
            # Get the latest commit SHA from the default branch
            commits_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/repository/commits/{default_branch}"
            commits_response = requests.get(commits_url, headers=headers, timeout=30)
            if commits_response.status_code == 200:
                ref = commits_response.json().get('id')
                print(f"Using commit {ref} from branch {default_branch} as reference")
            else:
                print(f"Warning: Could not get latest commit from {default_branch}. Release creation may fail.")
                ref = default_branch
        else:
            # Tag exists, use it as the ref
            ref = tag_name
            print(f"Tag {tag_name} exists, using it as reference")
    except Exception as e:
        print(f"Warning: Error checking tag existence: {e}")
        # Fallback to using the tag name as ref
        ref = tag_name

    print(f"Creating GitLab release {tag_name}...")
    release_data = {
        "tag_name": tag_name,
        "ref": ref,  # Add ref parameter pointing to the commit, branch or tag
        "name": release_name,
        "description": release_description
    }
    release_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases"
    
    if curl_with_retry(release_url, "POST", release_data):
        print(f"Successfully created or found release {tag_name}")
        return True
    else:
        print(f"ERROR: Failed to create release {tag_name}. Artifact linking might fail.")
        return False

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
    
    # Make sure file exists and is readable
    if not os.path.isfile(file_path):
        print(f"ERROR: File {file_path} does not exist or is not a file.")
        return False
    if not os.access(file_path, os.R_OK):
        print(f"ERROR: File {file_path} is not readable.")
        return False
    
    try:
        file_size = os.path.getsize(file_path)
        print(f"DEBUG: File size: {file_size} bytes")
        if file_size == 0:
            print(f"WARNING: File {filename} is empty. Skipping upload.")
            return False
    except OSError as e:
        print(f"ERROR: Cannot get size of file {file_path}: {e}")
        return False
    
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/uploads"
    response = curl_with_retry(upload_url, "POST", None, file_path)
    
    if response and isinstance(response, dict) and 'url' in response and 'alt' in response:
        print(f"Successfully uploaded file to project storage: {filename}")
        print(f"DEBUG: Upload response: {response}")
        if response.get('alt') != filename:
            print(f"WARNING: Uploaded filename '{response.get('alt')}' does not match original '{filename}'")
        return response
    else:
        print(f"ERROR: Failed to upload file {filename} to project storage or invalid response received.")
        if isinstance(response, bool):
            print("DEBUG: Upload call returned a boolean, expected a dictionary.")
        elif isinstance(response, dict):
            print(f"DEBUG: Received dictionary lacks expected keys 'url' or 'alt': {response}")
        else:
            print(f"DEBUG: Received unexpected response type: {type(response)}")
        return False

def check_existing_assets(gitlab_project_id, tag_name):
    """
    Check for existing assets in a release to avoid duplicates.
    
    :param gitlab_project_id: GitLab project ID
    :param tag_name: Release tag name
    :return: Dictionary of existing asset names mapped to their URLs
    """
    print(f"Checking for existing assets in release {tag_name}...")
    headers = {"PRIVATE-TOKEN": os.environ.get("GITLAB_TOKEN")}
    existing_assets = {}
    
    # Get release info
    release_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases/{tag_name}"
    try:
        response = requests.get(release_url, headers=headers, timeout=30)
        if response.status_code == 200:
            release_data = response.json()
            
            # Extract asset links
            if 'assets' in release_data and 'links' in release_data['assets']:
                for link in release_data['assets']['links']:
                    existing_assets[link['name']] = link['url']
                
            print(f"Found {len(existing_assets)} existing assets in the release")
            for name in existing_assets:
                print(f"  - {name}")
        else:
            print(f"Warning: Failed to get release info for {tag_name}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error checking existing assets: {e}")
        traceback.print_exc()
    
    return existing_assets

def add_file_as_release_asset(upload_info, gitlab_project_id, tag_name, project_path, existing_assets=None):
    """
    Add an uploaded file as a release asset.
    
    :param upload_info: Upload info from GitLab uploads API
    :param gitlab_project_id: GitLab project ID
    :param tag_name: Release tag name
    :param project_path: Project path for constructing correct download URLs
    :param existing_assets: Dictionary of existing asset names to avoid duplicates
    :return: True if successful, False otherwise
    """
    if not upload_info or not isinstance(upload_info, dict):
        print(f"ERROR: Invalid upload_info provided to add_file_as_release_asset: {upload_info}")
        return False

    uploaded_url_path = upload_info.get('url')
    asset_name = upload_info.get('alt')

    if not uploaded_url_path or not asset_name:
        print(f"ERROR: Missing 'url' or 'alt' in upload_info: {upload_info}")
        return False

    # Check if this asset already exists in the release
    if existing_assets and asset_name in existing_assets:
        print(f"SKIPPED: Asset '{asset_name}' already exists in the release. Skipping duplicate upload.")
        return True

    # Construct the public download URL for the release asset
    # Note: We still needed to upload the file first via upload_file_to_gitlab
    asset_link_url = f"https://gitlab-master.nvidia.com/{project_path}/-/releases/{tag_name}/downloads/{asset_name}"
    
    print(f"Adding file as release asset: {asset_name} using link URL: {asset_link_url}")
    
    link_data = {
        "name": asset_name,
        "url": asset_link_url,
        "link_type": "other"  # Use "other" for direct downloads
    }
    
    assets_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases/{tag_name}/assets/links"
    print(f"DEBUG: Posting to assets link URL: {assets_url}")
    print(f"DEBUG: Link data: {link_data}")
    
    if curl_with_retry(assets_url, "POST", link_data):
        print(f"Successfully added file {asset_name} as release asset link.")
        return True
    else:
        print(f"ERROR: Failed to add file {asset_name} as release asset link.")
        return False

def main():
    """Main function to upload APT packages to GitLab release as artifacts."""
    print("Starting GitLab APT package upload process (direct artifact mode)...")
    
    # Print script version for debugging
    print("Script version: 2.2.0 (APT direct artifacts with enhanced error handling)")
    
    # Print environment variables for debugging if DEBUG is set
    if os.environ.get("DEBUG"):
        print("DEBUG: Environment variables:")
        for key in ["GITLAB_PROJECT_ID", "TAG_NAME", "RELEASE_NAME", "RELEASE_DESCRIPTION", "APT_DIR"]:
            print(f"DEBUG: {key}={os.environ.get(key, 'Not set')}")
    
    try:
        # Validate required environment variables
        check_required_vars()
        
        # Test GitLab API connectivity 
        if not test_gitlab_connectivity():
            print("ERROR: Failed to connect to GitLab API. Exiting.")
            return 1
        
        # Check if the GitLab token has sufficient permissions
        if not check_gitlab_api_permissions():
            print("ERROR: GitLab token does not have sufficient permissions. Exiting.")
            return 1
        
        # Get environment variables
        gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
        tag_name = os.environ.get("TAG_NAME")
        apt_dir = os.environ.get("APT_DIR")
        gitlab_token = os.environ.get("GITLAB_TOKEN")

        # Fetch project path
        project_path = get_project_path(gitlab_project_id, gitlab_token)
        if not project_path:
            print("ERROR: Failed to fetch project path. Cannot construct correct download URLs.")
            return 1
        
        # Print system info
        print(f"DEBUG: Python version: {sys.version}")
        print(f"DEBUG: Requests version: {requests.__version__}")
        
        # Check for APT packages
        apt_files = check_apt_packages(apt_dir)
        
        if not apt_files:
            print("No APT packages found to upload. Exiting.")
            return 0 
        
        # Create a release in GitLab if it doesn't exist
        if not create_release():
            print("ERROR: Failed to create or find the release. Cannot proceed with artifact linking.")
            return 1
        
        # Get existing assets to avoid duplicates
        existing_assets = check_existing_assets(gitlab_project_id, tag_name)
        
        # Upload each APT file to GitLab and add as release asset
        upload_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"\n--- Processing {len(apt_files)} APT package(s) ---")
        for apt_file in apt_files:
            filename = os.path.basename(apt_file)
            print(f"\nProcessing APT package: {filename}")
            
            # Check if asset already exists
            if filename in existing_assets:
                print(f"Skipping {filename} - already exists in release {tag_name}")
                skipped_count += 1
                continue
                
            try:
                # Upload file to GitLab 
                upload_info = upload_file_to_gitlab(apt_file, gitlab_project_id)
                
                if upload_info:
                    # Add uploaded file as release asset, passing project_path
                    if add_file_as_release_asset(upload_info, gitlab_project_id, tag_name, project_path, existing_assets):
                        upload_count += 1
                        # Add to existing assets to avoid duplicate processing in this session
                        asset_name = upload_info.get('alt')
                        if asset_name:
                            existing_assets[asset_name] = "Added in this session"
                    else:
                        print(f"Failed to add {filename} as release asset after successful upload.")
                        failed_count += 1
                else:
                    print(f"Upload failed for {filename}. Skipping asset linking.")
                    failed_count += 1
            except Exception as e:
                print(f"ERROR: Unexpected exception processing APT package {filename}: {str(e)}")
                traceback.print_exc()
                failed_count += 1
            print(f"--- Finished processing {filename} ---")
        
        # Report results
        print("\n--- Upload Summary ---")
        print(f"- Total APT artifacts processed: {len(apt_files)}")
        print(f"- Successfully uploaded and linked: {upload_count}")
        print(f"- Skipped (already existing): {skipped_count}")
        print(f"- Failed: {failed_count}")
        
        if failed_count > 0:
            print("\nWARNING: One or more APT packages failed to upload or link. Check the logs above for details.")
            return 1 if upload_count == 0 and skipped_count == 0 else 0
        elif upload_count == 0 and skipped_count == 0 and len(apt_files) > 0:
            print("\nERROR: No APT packages were successfully uploaded, though some were found.")
            return 1
        else:
            print(f"\nAll found APT packages have been successfully processed for release {tag_name}.")
            return 0
            
    except Exception as e:
        print(f"FATAL: Unhandled exception in main execution: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as e:
        print(f"FATAL: Unhandled exception at script level: {str(e)}")
        traceback.print_exc()
    finally:
        print(f"Script finished with exit code {exit_code}")
        sys.exit(exit_code) 