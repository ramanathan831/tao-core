#!/usr/bin/env python3
import os
import sys
import glob
import time
import requests
import subprocess
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
    
    if not os.environ.get("WHEELS_DIR"):
        print("WARNING: WHEELS_DIR environment variable is not set, defaulting to 'third_party_wheels'")
        os.environ["WHEELS_DIR"] = "third_party_wheels"
    
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
        print(f"DEBUG: Testing connectivity to {url}")
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
        traceback.print_exc()
        return False

def convert_tarballs_to_wheels(wheels_dir):
    """Convert any .tar.gz files to wheels."""
    print(f"Checking for tarballs in {wheels_dir}...")
    
    # Ensure directory exists
    if not os.path.exists(wheels_dir):
        print(f"WARNING: Wheels directory {wheels_dir} does not exist")
        return
    
    # Find all .tar.gz files
    tarballs = glob.glob(f"{wheels_dir}/*.tar.gz")
    if not tarballs:
        print("No tarballs found, skipping conversion")
        return
    
    print(f"Found {len(tarballs)} tarballs to convert")
    
    for tarball in tarballs:
        try:
            print(f"Converting {os.path.basename(tarball)} to wheel...")
            # Try to build a wheel from the source distribution
            cmd = ["pip", "wheel", tarball, "-w", wheels_dir, "--no-deps"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully converted {os.path.basename(tarball)} to wheel")
                # Remove the source tarball to avoid upload issues
                os.remove(tarball)
                print(f"Removed source tarball {os.path.basename(tarball)}")
            else:
                print(f"Failed to build wheel from {os.path.basename(tarball)}, it will be skipped")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Exception converting {os.path.basename(tarball)}: {str(e)}")
            traceback.print_exc()

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
                    response = requests.request(method, url, headers=headers, files=files, timeout=180) # Increased timeout for uploads
            elif data:
                # With data case
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
                    # Return JSON if response is JSON, otherwise return True for success
                    if response.text and 'application/json' in response.headers.get('Content-Type', ''):
                        return response.json()
                    return True
                except ValueError as e:
                    print(f"WARNING: Failed to parse JSON response from {url} despite success code: {e}")
                    print(f"DEBUG: Full response text: {response.text}")
                    return True # Still a success, just not JSON
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
            # Don't retry on unexpected exceptions unless it's specifically coded for
            return False
    
    return False

def check_wheels(directory):
    """
    Check for wheel packages in the specified directory.
    
    :param directory: Directory to check for wheels
    :return: List of matching wheel files
    """
    if not os.path.isdir(directory):
        print(f"WARNING: Wheels directory '{directory}' does not exist")
        return []
    
    # Count files
    files = glob.glob(f"{directory}/*.whl")
    file_count = len(files)
    
    if file_count == 0:
        print(f"WARNING: No wheel files found in {directory}.")
    else:
        print(f"Found {file_count} wheel files in {directory} to upload.")
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

    # First check if the release already exists
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
            # Proceed with creation attempt anyway

    except requests.exceptions.RequestException as e:
        print(f"Error checking for existing release {tag_name}: {e}")
        print("Proceeding with creation attempt...")

    print(f"Creating GitLab release {tag_name}...")
    release_data = {
        "tag_name": tag_name,
        "name": release_name,
        "description": release_description
    }
    release_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/releases"
    
    if curl_with_retry(release_url, "POST", release_data):
        print(f"Successfully created or found release {tag_name}")
        return True
    else:
        print(f"ERROR: Failed to create release {tag_name}. Artifact linking might fail.")
        return False # Indicate failure

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
            return False # Don't upload empty files
    except OSError as e:
        print(f"ERROR: Cannot get size of file {file_path}: {e}")
        return False
    
    upload_url = f"https://gitlab-master.nvidia.com/api/v4/projects/{gitlab_project_id}/uploads"
    response = curl_with_retry(upload_url, "POST", None, file_path)
    
    # Check if the response looks like a valid upload response (should be JSON dict)
    if response and isinstance(response, dict) and 'url' in response and 'alt' in response:
        print(f"Successfully uploaded file to project storage: {filename}")
        print(f"DEBUG: Upload response: {response}")
        # Verify the 'alt' field matches the filename
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

def add_file_as_release_asset(upload_info, gitlab_project_id, tag_name):
    """
    Add an uploaded file as a release asset.
    
    :param upload_info: Upload info from GitLab uploads API
    :param gitlab_project_id: GitLab project ID
    :param tag_name: Release tag name
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

    # Construct the full URL for the asset link
    asset_link_url = f"https://gitlab-master.nvidia.com/{gitlab_project_id}/-/uploads/{os.path.basename(uploaded_url_path)}"
    # asset_link_url = f"https://gitlab-master.nvidia.com{uploaded_url_path}" 
    
    print(f"Adding file as release asset: {asset_name} using link URL: {asset_link_url}")
    
    link_data = {
        "name": asset_name,
        "url": asset_link_url,
        # "filepath": uploaded_url_path, # Sometimes filepath is needed instead of url
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
        # Optionally, try with filepath if url failed?
        # link_data_alt = {"name": asset_name, "filepath": uploaded_url_path, "link_type": "other"}
        # print("DEBUG: Retrying asset link creation with filepath...")
        # if curl_with_retry(assets_url, "POST", link_data_alt):
        #     print(f"Successfully added file {asset_name} using filepath.")
        #     return True
        # else:
        #     print(f"ERROR: Failed to add file {asset_name} using filepath as well.")
        #     return False
        return False

def main():
    """Main function to upload Python wheels to GitLab release as artifacts."""
    print("Starting GitLab wheel package upload process (direct artifact)...")
    
    # Print environment variables for debugging if DEBUG is set
    if os.environ.get("DEBUG"):
        print("DEBUG: Environment variables:")
        for key in ["GITLAB_PROJECT_ID", "TAG_NAME", "RELEASE_NAME", "RELEASE_DESCRIPTION", "WHEELS_DIR"]:
            print(f"DEBUG: {key}={os.environ.get(key, 'Not set')}")
    
    try:
        # Validate required environment variables
        check_required_vars()
        
        # Test GitLab API connectivity first
        if not test_gitlab_connectivity():
            print("ERROR: Failed to connect to GitLab API. Exiting.")
            return 1
        
        # Get environment variables
        gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID")
        tag_name = os.environ.get("TAG_NAME")
        wheels_dir = os.environ.get("WHEELS_DIR")
        
        # Print system info
        print(f"DEBUG: Python version: {sys.version}")
        print(f"DEBUG: Requests version: {requests.__version__}")
        
        # Convert any tarballs to wheels first
        convert_tarballs_to_wheels(wheels_dir)
        
        # Check for wheel packages
        wheel_files = check_wheels(wheels_dir)
        
        if not wheel_files:
            print("No wheel packages found to upload in {wheels_dir}. Exiting.")
            # Decide if this is an error or just no work to do.
            # Assuming it's okay if no wheels are present.
            print("No wheels found, exiting successfully.")
            return 0 
        
        # Create a release in GitLab if it doesn't exist
        if not create_release():
             # If release creation failed, we cannot link artifacts
             print("ERROR: Failed to create or find the release. Cannot proceed with artifact linking.")
             return 1
        
        # Upload each wheel file to GitLab and add as release asset
        upload_count = 0
        failed_count = 0
        
        print(f"\n--- Processing {len(wheel_files)} wheel file(s) ---")
        for wheel in wheel_files:
            filename = os.path.basename(wheel)
            print(f"\nProcessing wheel: {filename}")
            
            try:
                # Upload file to GitLab first
                upload_info = upload_file_to_gitlab(wheel, gitlab_project_id)
                
                if upload_info:
                    # Add uploaded file as release asset
                    if add_file_as_release_asset(upload_info, gitlab_project_id, tag_name):
                        upload_count += 1
                    else:
                        print(f"Failed to add {filename} as release asset after successful upload.")
                        failed_count += 1
                else:
                    # Upload failed
                    print(f"Upload failed for {filename}. Skipping asset linking.")
                    failed_count += 1
            except Exception as e:
                print(f"ERROR: Unexpected exception processing wheel {filename}: {str(e)}")
                traceback.print_exc()
                failed_count += 1
            print(f"--- Finished processing {filename} ---")
        
        # Report results
        print("\n--- Upload Summary ---")
        print(f"- Total wheel artifacts processed: {len(wheel_files)}")
        print(f"- Successfully uploaded and linked: {upload_count}")
        print(f"- Failed: {failed_count}")
        
        if failed_count > 0:
            print("\nWARNING: One or more wheels failed to upload or link. Check the logs above for details.")
            # Return success if at least some uploads worked, otherwise fail
            return 1 if upload_count == 0 else 0
        elif upload_count == 0 and len(wheel_files) > 0:
             print("\nERROR: No wheels were successfully uploaded, though some were found.")
             return 1
        else:
            print("\nAll found wheels have been successfully uploaded as release artifacts.")
            return 0
            
    except Exception as e:
        print(f"FATAL: Unhandled exception in main execution: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = 1 # Default to failure
    try:
        exit_code = main()
    except Exception as e:
        print(f"FATAL: Unhandled exception at script level: {str(e)}")
        traceback.print_exc()
    finally:
        print(f"Script finished with exit code {exit_code}")
        sys.exit(exit_code) 