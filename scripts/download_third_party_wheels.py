#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import glob
import shutil
import argparse

def download_with_retry(requirements_file):
    """Attempt to download binary wheels with retries."""
    max_attempts = 3
    attempt = 1
    delay = 5
    
    while attempt <= max_attempts:
        print(f"Attempt {attempt} of {max_attempts}: Downloading binary wheels from {requirements_file}...")
        try:
            subprocess.run(
                ["pip", "download", "--only-binary=:all:", "-r", requirements_file, 
                 "-d", "third_party_wheels", "--no-deps"],
                check=True
            )
            print("Successfully downloaded all binary wheels!")
            return True
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
            delay *= 2
    
    print(f"Could not download all wheels as binaries after {max_attempts} attempts.")
    return False

def process_individual_packages(requirements_file):
    """Process packages individually from requirements file."""
    print(f"Falling back to downloading packages individually from {requirements_file}...")
    
    with open(requirements_file, "r") as req_file:
        for package in req_file:
            package = package.strip()
            
            # Skip comments and empty lines
            if not package or package.startswith("#"):
                continue
            
            print(f"Processing package: {package}")
            
            # Try multiple methods for each package
            try:
                # Method 1: Try binary wheel
                subprocess.run(
                    ["pip", "download", "--only-binary=:all:", package, 
                     "-d", "third_party_wheels", "--no-deps"],
                    check=True
                )
                print(f"Successfully downloaded binary wheel for {package}")
            except subprocess.CalledProcessError:
                try:
                    # Method 2: Try binary wheel from PyPI
                    subprocess.run(
                        ["pip", "download", "--only-binary=:all:", package, 
                         "-d", "third_party_wheels", "--no-deps", 
                         "--extra-index-url", "https://pypi.org/simple"],
                        check=True
                    )
                    print(f"Successfully downloaded binary wheel for {package} from PyPI")
                except subprocess.CalledProcessError:
                    try:
                        # Method 3: Try source distribution
                        subprocess.run(
                            ["pip", "download", "--no-binary=:none:", package, 
                             "-d", "third_party_wheels", "--no-deps"],
                            check=True
                        )
                        print(f"Warning: Could only download source distribution for {package}")
                        
                        # If only got a source distribution, try to build a wheel
                        source_files = glob.glob(f"third_party_wheels/*{package}*.tar.gz")
                        for src in source_files:
                            if os.path.isfile(src):
                                print(f"Attempting to build wheel from source for {package}...")
                                try:
                                    subprocess.run(
                                        ["pip", "wheel", src, "-w", "third_party_wheels", "--no-deps"],
                                        check=True
                                    )
                                    # Check if wheel was successfully built
                                    wheel_files = glob.glob(f"third_party_wheels/*{package}*.whl")
                                    if wheel_files:
                                        # Clean up source distribution
                                        os.remove(src)
                                except subprocess.CalledProcessError:
                                    print(f"Failed to build wheel for {package}, keeping source distribution")
                    except subprocess.CalledProcessError:
                        print(f"Warning: Failed to download {package}, continuing with other packages")

def main():
    """Main function to download third-party wheels."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download Python wheels listed in a requirements file")
    parser.add_argument("--requirements", "-r", default="requirements.txt", 
                        help="Path to Python requirements file (default: requirements.txt)")
    args = parser.parse_args()
    
    # Check if requirements file exists
    if not os.path.isfile(args.requirements):
        print(f"Error: Requirements file '{args.requirements}' not found")
        return 1
        
    # Create output directory for third party wheels
    os.makedirs("third_party_wheels", exist_ok=True)
    
    print(f"Starting download of third-party wheels from {args.requirements}...")
    
    # Try to download binary wheels
    if not download_with_retry(args.requirements):
        process_individual_packages(args.requirements)
    
    # Check what we've downloaded
    print("Summary of downloaded wheels:")
    wheels = glob.glob("third_party_wheels/*")
    for wheel in wheels:
        print(f"  {os.path.basename(wheel)}")
    
    print("Third-party wheel download process completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 