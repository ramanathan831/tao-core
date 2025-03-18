#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import glob
import shutil
import argparse

def setup_apt_directory():
    """Create and setup apt packages directory."""
    apt_dir = "third_party_apt"
    os.makedirs(apt_dir, exist_ok=True)
    return apt_dir

def setup_apt_sources_directory():
    """Create and setup apt sources directory for license extraction."""
    apt_sources_dir = "third_party_apt_sources"
    os.makedirs(apt_sources_dir, exist_ok=True)
    return apt_sources_dir

def download_apt_package(package_name, apt_dir, retries=3, delay=5):
    """
    Download a single apt package and its dependencies.
    
    :param package_name: Name of the apt package to download
    :param apt_dir: Directory to save the downloaded packages
    :param retries: Number of retry attempts
    :param delay: Delay between retries in seconds
    :return: True if successful, False otherwise
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} of {retries}: Downloading apt package {package_name}")
            
            # Download the package without installing it
            subprocess.run(
                ["apt-get", "download", package_name],
                cwd=apt_dir,
                check=True
            )
            print(f"Successfully downloaded apt package: {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to download apt package {package_name}: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Max retries reached for {package_name}")
                return False

def download_apt_source(package_name, sources_dir, retries=3, delay=5):
    """
    Download the source code of an apt package for license extraction.
    
    :param package_name: Name of the apt package to download source for
    :param sources_dir: Directory to save the source packages
    :param retries: Number of retry attempts
    :param delay: Delay between retries in seconds
    :return: True if successful, False otherwise
    """
    package_source_dir = os.path.join(sources_dir, package_name)
    os.makedirs(package_source_dir, exist_ok=True)
    
    current_dir = os.getcwd()
    
    try:
        # Change to package source directory
        os.chdir(package_source_dir)
        
        for attempt in range(1, retries + 1):
            try:
                print(f"Attempt {attempt} of {retries}: Downloading source for apt package {package_name}")
                
                # Use apt-get source to get the package source
                result = subprocess.run(
                    ["apt-get", "source", package_name],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Successfully downloaded source for apt package: {package_name}")
                
                # Extract the downloaded directory name from output
                source_dir = None
                for line in result.stdout.split('\n'):
                    if line.startswith('Unpacking ') and ' (' in line:
                        source_dir = line.split('Unpacking ')[1].split(' (')[0]
                        break
                
                # Create a marker file indicating success
                with open(".source_download_successful", "w") as marker:
                    marker.write(source_dir if source_dir else "unknown")
                
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed to download source for apt package {package_name}: {e}")
                print(f"Error output: {e.stderr}")
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"Max retries reached for downloading source of {package_name}")
                    return False
    finally:
        # Return to the original directory
        os.chdir(current_dir)

def update_apt_cache(retries=3, delay=5):
    """Update apt cache to ensure we get the latest package information."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} of {retries}: Updating apt cache")
            subprocess.run(["apt-get", "update"], check=True)
            print("Successfully updated apt cache")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to update apt cache: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Max retries reached for apt-get update")
                return False

def main():
    """Main function to download apt packages."""
    parser = argparse.ArgumentParser(description="Download apt packages listed in a requirements file")
    parser.add_argument("--requirements", "-r", default="apt-requirements.txt", 
                        help="Path to apt requirements file (default: apt-requirements.txt)")
    parser.add_argument("--download-sources", "-s", action="store_true",
                        help="Download package sources for license extraction (default: False)")
    args = parser.parse_args()
    
    # Setup apt directories
    apt_dir = setup_apt_directory()
    apt_sources_dir = setup_apt_sources_directory() if args.download_sources else None
    
    # Check if requirements file exists
    if not os.path.isfile(args.requirements):
        print(f"Error: Requirements file '{args.requirements}' not found")
        return 1
    
    # Update apt cache first
    if not update_apt_cache():
        print("Failed to update apt cache. Continuing with download attempts...")
    
    # Read and process packages from requirements file
    successful_packages = []
    failed_packages = []
    successful_sources = []
    failed_sources = []
    
    with open(args.requirements, "r") as req_file:
        for line in req_file:
            package_name = line.strip()
            
            # Skip empty lines and comments
            if not package_name or package_name.startswith("#"):
                continue
            
            print(f"Processing apt package: {package_name}")
            if download_apt_package(package_name, apt_dir):
                successful_packages.append(package_name)
            else:
                failed_packages.append(package_name)
            
            # Download source if requested
            if args.download_sources:
                print(f"Downloading source for apt package: {package_name}")
                if download_apt_source(package_name, apt_sources_dir):
                    successful_sources.append(package_name)
                else:
                    failed_sources.append(package_name)
    
    # Print summary for binary packages
    print("\nBinary Packages Download Summary:")
    print(f"Successfully downloaded {len(successful_packages)} packages")
    print(f"Failed to download {len(failed_packages)} packages")
    
    if failed_packages:
        print("\nFailed binary packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
    
    # Print summary for source packages if requested
    if args.download_sources:
        print("\nSource Packages Download Summary:")
        print(f"Successfully downloaded {len(successful_sources)} package sources")
        print(f"Failed to download {len(failed_sources)} package sources")
        
        if failed_sources:
            print("\nFailed source packages:")
            for pkg in failed_sources:
                print(f"  - {pkg}")
    
    # Write package list to a manifest file
    with open(os.path.join(apt_dir, "packages.manifest"), "w") as manifest:
        manifest.write("# Successfully downloaded packages\n")
        for pkg in successful_packages:
            manifest.write(f"{pkg}\n")
        
        if failed_packages:
            manifest.write("\n# Failed packages\n")
            for pkg in failed_packages:
                manifest.write(f"# {pkg}\n")
    
    # Write source package list to a manifest file if sources were downloaded
    if args.download_sources:
        with open(os.path.join(apt_sources_dir, "sources.manifest"), "w") as manifest:
            manifest.write("# Successfully downloaded source packages\n")
            for pkg in successful_sources:
                manifest.write(f"{pkg}\n")
            
            if failed_sources:
                manifest.write("\n# Failed source packages\n")
                for pkg in failed_sources:
                    manifest.write(f"# {pkg}\n")
    
    return 0 if not (failed_packages or (args.download_sources and failed_sources)) else 1

if __name__ == "__main__":
    sys.exit(main()) 