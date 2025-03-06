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

"""Dataset upload modules"""
import tarfile
import os
import glob
import sys

from handlers.cloud_storage import create_cs_instance
from utils import read_network_config


# Simple helper class for ease of code migration
class SimpleHandler:
    """Helper class holding dataset information"""

    def __init__(self, org_name, handler_metadata, temp_dir="", workspace_metadata=None):
        """Initialize the Handler helper class"""
        self.root = temp_dir
        self.type = handler_metadata.get("type")
        self.format = handler_metadata.get("format")
        self.intent = handler_metadata.get("use_for", [])
        assert type(self.intent) is list
        self.cloud_instance = None
        if workspace_metadata:
            self.cloud_instance, _ = create_cs_instance(workspace_metadata)

    def check_for_file_existence(self, path, file_type="file"):
        """Check for existence of file"""
        if self.cloud_instance:
            if file_type == "file":
                return self.cloud_instance.is_file(path)
            path = path[1:] if path.startswith("/") else path
            return self.cloud_instance.is_folder(path)
        return os.path.exists(path)


def _untar_file(tar_path, dest, strip_components=0):
    """Function to untar a file"""
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            # Remove leading directory components using strip_components
            components = member.name.split(os.sep)
            if len(components) > strip_components:
                member.name = os.path.join(*components[strip_components:])
            if member.isdir():
                # Make subdirs ahead because tarfile extracts them with user permissions only
                os.makedirs(os.path.join(dest, member.name), exist_ok=True)
            tar.extract(member, path=dest, set_attrs=False)


def _extract_images(tar_path, dest):
    """Function to extract images, other directories on same level as images to root of dataset"""
    # Infer how many components to strip to get images,labels to top of dataset directory
    # Assumes: images, other necessary directories are in the same level
    with tarfile.open(tar_path) as tar:
        strip_components = 0
        names = [tinfo.name for tinfo in tar.getmembers()]
        for name in names:
            if "/images/" in name:
                strip_components = name.split("/").index("images")
                break
    # Build shell command for untarring
    print("Untarring data started", file=sys.stderr)
    _untar_file(tar_path, dest, strip_components)
    print("Untarring data complete", file=sys.stderr)

    # Remove .tar.gz file
    print("Removing data tar file", file=sys.stderr)
    os.remove(tar_path)
    print("Deleted data tar file", file=sys.stderr)


def write_dir_contents(directory, file):
    """Write contents of a directory to a file"""
    with open(file, "w", encoding='utf-8') as f:
        for dir_files in sorted(glob.glob(directory + "/*")):
            f.write(dir_files + "\n")


def validate_dataset(org_name, handler_metadata, temp_dir="", workspace_metadata=None):
    """Generic dataset validator using config"""
    handler = SimpleHandler(org_name, handler_metadata, temp_dir=temp_dir, workspace_metadata=workspace_metadata)

    try:
        # Load network config
        print("handler.type", handler.type, file=sys.stderr)
        network_config = read_network_config(handler.type)
        print("network_config", network_config, file=sys.stderr)
        validation_config = network_config.get("dataset_validation", {})

        # Get format-specific requirements, fallback to default
        format_reqs = validation_config.get("required_files", {}).get(
            handler.format,
            validation_config.get("required_files", {}).get("default", [])
        )

        # Validate each requirement
        for req in format_reqs:
            if "path" in req:
                path = os.path.join(handler.root, req["path"])
                file_type = req.get("type", "file")
                error_msg = f"Required file not found: {path}"
                assert handler.check_for_file_existence(path, file_type=file_type), error_msg
            elif "all_of" in req:
                # Check if all requirements are met
                for subreq in req["all_of"]:
                    path = os.path.join(handler.root, subreq["path"])
                    file_type = subreq.get("type", "file")
                    error_msg = f"Required file not found: {path}"
                    assert handler.check_for_file_existence(path, file_type=file_type), error_msg
            elif "any_of" in req:
                # Check if any of the requirements are met
                any_valid = False
                for subreq in req["any_of"]:
                    if "path" in subreq:
                        path = os.path.join(handler.root, subreq["path"])
                        file_type = subreq.get("type", "file")
                        if handler.check_for_file_existence(path, file_type=file_type):
                            any_valid = True
                            break
                    elif "all_of" in subreq:
                        # Check if all sub-requirements are met
                        all_valid = True
                        for subsubreq in subreq["all_of"]:
                            path = os.path.join(handler.root, subsubreq["path"])
                            file_type = subsubreq.get("type", "file")
                            if not handler.check_for_file_existence(path, file_type=file_type):
                                all_valid = False
                                break
                        if all_valid:
                            any_valid = True
                            break
                assert any_valid, f"None of the alternative requirements are met: {req['any_of']}"
            elif "intent_based_path" in req:
                # Check if intent exists
                assert handler.intent, "Intent is required for this dataset"
                assert len(handler.intent) == 1, "Only one intent is allowed"
                intent = handler.intent[0]

                # Get path requirement for this intent
                intent_req = req["intent_based_path"].get(intent)
                assert intent_req, f"No path requirement found for intent: {intent}"

                path = os.path.join(handler.root, intent_req["path"])
                file_type = intent_req.get("type", "file")
                error_msg = f"Required file not found: {path}"
                assert handler.check_for_file_existence(path, file_type=file_type), error_msg
            if "intent_restriction" in req:
                if handler.intent:
                    assert handler.intent == req["intent_restriction"]

        return True

    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        return False
