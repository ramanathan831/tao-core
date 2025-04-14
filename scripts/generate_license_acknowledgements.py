#!/usr/bin/env python3
import os
import tempfile
import shutil
import tarfile
import zipfile
import requests
import glob
import re
import argparse

# Templates for the license file
license_file_text = """
# Acknowledgements

Third party packages and their licenses

{license_string}
"""

package_template = """
## {package_title}


```text
{license_body}
```

"""

def get_package_info_from_wheel(wheel_path):
    """
    Extract package name and version from wheel filename.
    
    :param wheel_path: Path to the wheel file
    :return: Tuple of (package_name, version)
    """
    filename = os.path.basename(wheel_path)
    parts = filename.split('-')
    
    if len(parts) >= 3:
        package_name = parts[0]
        version = parts[1]
        return package_name, version
    else:
        # Handle special cases or malformed filenames
        print(f"Warning: Could not parse package info from {filename}")
        return filename.replace('.whl', ''), None

def get_pypi_package_license(package_name, version=None):
    """
    Downloads a package from PyPI and extracts its license information.
    
    :param package_name: Name of the package to download.
    :param version: Specific version of the package (optional).
    :return: License text if found, otherwise None.
    """
    try:
        # Create a temporary directory to store the package
        temp_dir = tempfile.mkdtemp()
        # Construct the PyPI URL for the package
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        response.raise_for_status()
        package_info = response.json()
        
        # Get the release URL
        release = package_info["releases"][version] if version else package_info["urls"]

        if not release:
            print(f"No releases found for package {package_name}")
            return None
        
        # Get the first release file URL
        release_url = release[0]["url"]
        response = requests.get(release_url, stream=True)
        response.raise_for_status()

        file_path = os.path.join(temp_dir, release_url.split("/")[-1])

        # Save the downloaded file
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # Extract the package
        extracted_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        if file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(extracted_dir)
        elif file_path.endswith(".zip") or file_path.endswith(".whl"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extracted_dir)
        else:
            raise NotImplementedError("Unsupported file format.")

        # Look for license files
        license_file = None
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                if "LICENSE" in file.upper() or "LICENCE" in file.upper():
                    license_file = os.path.join(root, file)
                    break

        if license_file:
            with open(license_file, "r", encoding="utf-8", errors="replace") as f:
                license_text = f.read()
            return license_text
        else:
            print(f"License file not found for {package_name}")
            return None

    except Exception as e:
        print(f"An error occurred with {package_name}: {e}")
        return None
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

def extract_license_from_local_wheel(wheel_path):
    """
    Attempt to extract license information from a local wheel file.
    
    :param wheel_path: Path to the wheel file
    :return: License text if found, otherwise None
    """
    try:
        temp_dir = tempfile.mkdtemp()
        extracted_dir = os.path.join(temp_dir, "extracted")
        
        # Extract the wheel
        with zipfile.ZipFile(wheel_path, "r") as zip_ref:
            zip_ref.extractall(extracted_dir)
            
        # Look for license files in the extracted wheel
        license_file = None
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                if "LICENSE" in file.upper() or "LICENCE" in file.upper():
                    license_file = os.path.join(root, file)
                    break
                    
        if license_file:
            with open(license_file, "r", encoding="utf-8", errors="replace") as f:
                license_text = f.read()
            return license_text
        else:
            print(f"No license file found in {wheel_path}")
            return None
            
    except Exception as e:
        print(f"Error extracting license from wheel {wheel_path}: {e}")
        return None
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def extract_license_from_apt_source(package_name):
    """
    Extract license information from a downloaded APT source package.
    
    :param package_name: Name of the APT package
    :return: License text if found, otherwise None
    """
    source_dir = os.path.join("third_party_apt_sources", package_name)
    
    if not os.path.isdir(source_dir):
        print(f"No source directory found for APT package {package_name}")
        return None
    
    # Check if download was successful by looking for marker file
    marker_file = os.path.join(source_dir, ".source_download_successful")
    if not os.path.exists(marker_file):
        print(f"Source download was not successful for APT package {package_name}")
        return None
    
    # Look for license files in the source package directory
    common_license_filenames = [
        "LICENSE", "License", "license",
        "LICENCE", "Licence", "licence",
        "COPYING", "Copying", "copying",
        "COPYRIGHT", "Copyright", "copyright"
    ]
    
    license_files = []
    
    # Walk through the source directory to find license files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(license_name in file for license_name in common_license_filenames):
                license_files.append(os.path.join(root, file))
    
    if not license_files:
        print(f"No license files found for APT package {package_name}")
        return None
    
    # Combine all found license files into a single text
    license_text = ""
    for license_file in license_files:
        try:
            with open(license_file, "r", encoding="utf-8", errors="replace") as f:
                license_text += f"=== {os.path.basename(license_file)} ===\n\n"
                license_text += f.read()
                license_text += "\n\n"
        except Exception as e:
            print(f"Error reading license file {license_file}: {e}")
    
    return license_text.strip() if license_text else None

def main():
    """Generate acknowledgements.md file from third-party wheels and APT packages."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate license acknowledgements for third-party packages")
    parser.add_argument("--include-apt", "-a", action="store_true", 
                        help="Include APT package licenses (default: False)")
    parser.add_argument("--output", "-o", default="acknowledgements.md", 
                        help="Output filename (default: acknowledgements.md)")
    args = parser.parse_args()
    
    wheels_dir = "third_party_wheels"
    
    license_dictionary = {}
    missed_packages = []
    
    # Process Python wheels
    if os.path.isdir(wheels_dir):
        # Find all wheel files
        wheel_files = glob.glob(f"{wheels_dir}/*.whl")
        if wheel_files:
            print(f"Found {len(wheel_files)} wheel files. Processing licenses...")
            
            for wheel_path in wheel_files:
                package_name, version = get_package_info_from_wheel(wheel_path)
                print(f"Processing Python package: {package_name} (version: {version})")
                
                # First try to extract license from the local wheel
                license_text = extract_license_from_local_wheel(wheel_path)
                
                # If that fails, try to get it from PyPI
                if not license_text:
                    print(f"No license found in wheel, trying PyPI for {package_name}")
                    license_text = get_pypi_package_license(package_name, version)
                    
                if license_text:
                    license_dictionary[f"python:{package_name}"] = license_text
                else:
                    missed_packages.append(f"python:{package_name}")
        else:
            print(f"No wheel files found in {wheels_dir}")
    else:
        print(f"Wheels directory {wheels_dir} does not exist")
    
    # Process APT packages if requested
    if args.include_apt:
        apt_dir = "third_party_apt"
        apt_sources_dir = "third_party_apt_sources"
        
        if os.path.isdir(apt_dir) and os.path.isdir(apt_sources_dir):
            # Get list of apt packages from manifest file
            manifest_path = os.path.join(apt_dir, "packages.manifest")
            if os.path.isfile(manifest_path):
                print("Processing APT package licenses...")
                
                with open(manifest_path, "r") as manifest_file:
                    for line in manifest_file:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith("#"):
                            continue
                        
                        package_name = line
                        print(f"Processing APT package: {package_name}")
                        
                        # Try to extract license from source
                        license_text = extract_license_from_apt_source(package_name)
                        
                        if license_text:
                            license_dictionary[f"apt:{package_name}"] = license_text
                        else:
                            missed_packages.append(f"apt:{package_name}")
            else:
                print(f"APT packages manifest file not found: {manifest_path}")
        else:
            print(f"APT directories not found: {apt_dir} or {apt_sources_dir}")
    
    # Generate the acknowledgements.md file
    if license_dictionary:
        license_body = ""
        for package_key, package_license in license_dictionary.items():
            # Extract the actual package name from the key (format: "type:name")
            package_type, package_name = package_key.split(":", 1)
            package_title = f"{package_name.capitalize()} ({package_type})"
            
            license_body += package_template.format(
                package_title=package_title,
                license_body=package_license
            )
            
        with open(os.path.join(os.getcwd(), args.output), "w", encoding="utf-8") as ackfile:
            ackfile.write(license_file_text.format(license_string=license_body))
            
        # Report results
        print(f"Generated {args.output} with licenses for {len(license_dictionary)} packages")
    else:
        print("No license information found for any packages")
    
    if missed_packages:
        print(f"Could not find licenses for {len(missed_packages)} packages:")
        for pkg in missed_packages:
            print(f"  - {pkg}")
            
    return 0
    
if __name__ == "__main__":
    import sys
    sys.exit(main()) 