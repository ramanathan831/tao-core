# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Cloud storage fsspec client"""
import os
import copy
import time
import json
import fsspec
import logging
import functools
from datetime import datetime

from nvidia_tao_core.microservices.handlers.encrypt import NVVaultEncryption
from nvidia_tao_core.distributed.decorators import master_node_only

NUM_RETRY = 5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retry_method(func):
    """Retry Cloud storage methods for NUM_RETRY times"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for _ in range(NUM_RETRY):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log or handle the exception as needed
                logger.error("Exception in %s: %s", func.__name__, e)
            time.sleep(30)
        # If all retries fail, raise an exception or handle it accordingly
        raise ValueError(f"Failed to execute {func.__name__} after multiple retries")
    return wrapper


def create_cs_instance_with_decrypted_metadata(decrypted_metadata):
    """Create a Cloud storage instance based on decrypted metadata"""
    handler_metadata_copy = copy.deepcopy(decrypted_metadata)
    cloud_type = handler_metadata_copy.get("cloud_type")
    cloud_specific_details = handler_metadata_copy.get("cloud_specific_details")
    cloud_bucket_name = cloud_specific_details.get("cloud_bucket_name")

    cs_instance = None
    if cloud_specific_details and cloud_bucket_name:
        if cloud_type == "aws":
            cs_instance = CloudStorage(
                cloud_type="aws",
                bucket_name=cloud_bucket_name,
                key=cloud_specific_details.get("access_key"),
                secret=cloud_specific_details.get("secret_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )

        elif cloud_type == "azure":
            cs_instance = CloudStorage(
                cloud_type="azure",
                bucket_name=cloud_bucket_name,
                key=cloud_specific_details.get("account_name"),
                secret=cloud_specific_details.get("access_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )

    return cs_instance, cloud_specific_details


def create_cs_instance(handler_metadata):
    """Create a Cloud Storage instance based on handler metadata details"""
    handler_metadata_copy = copy.deepcopy(handler_metadata)
    cloud_type = handler_metadata_copy.get("cloud_type")
    cloud_specific_details = handler_metadata_copy.get("cloud_specific_details")

    # Decrypt cloud details
    config_path = os.getenv("VAULT_SECRET_PATH", None)
    if config_path:
        encryption = NVVaultEncryption(config_path)
        for key, encrypted_value in cloud_specific_details.items():
            if encryption.check_config()[0]:
                cloud_specific_details[key] = encryption.decrypt(encrypted_value)

    cloud_bucket_name = cloud_specific_details.get("cloud_bucket_name")

    cs_instance = None
    if cloud_specific_details:
        if cloud_type == "aws":
            cs_instance = CloudStorage(
                cloud_type="aws",
                bucket_name=cloud_bucket_name,
                key=cloud_specific_details.get("access_key"),
                secret=cloud_specific_details.get("secret_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
        elif cloud_type == "azure":
            cs_instance = CloudStorage(
                cloud_type="azure",
                bucket_name=cloud_bucket_name,
                key=cloud_specific_details.get("account_name"),
                secret=cloud_specific_details.get("access_key"),
                client_kwargs={"endpoint_url": cloud_specific_details.get("endpoint_url")}
            )
    return cs_instance, cloud_specific_details


def get_date_time():
    """Get the current date and time"""
    date_time = datetime.now()
    date_object = date_time.date()
    time_object = date_time.time()
    time_data = "{}:{}:{}".format(  # noqa pylint: disable=C0209
        time_object.hour,
        time_object.minute,
        time_object.second
    )
    date_data = "{}/{}/{}".format(  # noqa pylint: disable=C0209
        date_object.month,
        date_object.day,
        date_object.year
    )
    return date_data, time_data


class CloudStorage:
    """Cloud storage CRUD operations using fsspec for S3 and Azure."""

    @retry_method
    def __init__(self, cloud_type, bucket_name, **kwargs):
        """Initialize the CloudStorageFSSpec object.

        cloud_type: 'aws' or 'azure'.
        bucket_name: Name of the bucket/container.
        kwargs: Additional arguments for fsspec.filesystem (e.g., key, secret, client_kwargs, etc.)
        """
        self.cloud_type = cloud_type
        self.bucket_name = bucket_name
        self.fs = None
        if cloud_type == 'aws':
            self.fs = fsspec.filesystem('s3', **kwargs)
            self.root = f'{bucket_name}/'
        elif cloud_type == 'azure':
            self.fs = fsspec.filesystem('az', **kwargs)
            self.root = f'{bucket_name}/'
        else:
            raise ValueError("Unsupported cloud_type. Use 'aws' or 'azure'.")

    @retry_method
    def is_file(self, cloud_path):
        """Check if the given cloud path is a file."""
        full_path = self.root + cloud_path
        try:
            return self.fs.isfile(full_path)
        except Exception as e:
            logger.error(f"is_file error: {e}")
            return False

    @retry_method
    def is_folder(self, cloud_path):
        """Check if the given cloud path is a folder."""
        full_path = self.root + cloud_path.rstrip('/') + '/'
        try:
            return self.fs.isdir(full_path)
        except Exception as e:
            logger.error(f"is_folder error: {e}")
            return False

    @retry_method
    def glob_files(self, pattern):
        """Return a list of files matching the pattern."""
        full_pattern = self.root + pattern
        try:
            return [p[len(self.root):] for p in self.fs.glob(full_pattern) if self.fs.isfile(p)]
        except Exception as e:
            logger.error(f"glob_files error: {e}")
            return []

    @retry_method
    def list_files_in_folder(self, folder):
        """Recursively list files in the specified folder and its subfolders."""
        full_path = self.root + folder.rstrip('/') + '/'
        try:
            all_paths = self.fs.find(full_path)
            file_names = [p[len(self.root):] for p in all_paths if self.fs.isfile(p)]
            return file_names, []  # Details not available recursively
        except Exception as e:
            logger.error(f"list_files_in_folder error: {e}")
            return [], []

    @retry_method
    def download_file(self, cloud_file_path, local_destination):
        """Download a file from cloud to local."""
        logger.info(f"Starting download of file: {cloud_file_path} to {local_destination}")
        full_path = self.root + cloud_file_path
        try:
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)
            self.fs.get(full_path, local_destination)
        except Exception as e:
            logger.error(f"download_file error: {e}")
        logger.info(f"Successfully downloaded file: {cloud_file_path} to {local_destination}")

    @retry_method
    def download_folder(self, cloud_folder, local_destination, maintain_src_folder_structure=False):
        """Download all files in a cloud folder to a local destination."""
        from nvidia_tao_core.cloud_handlers.utils import status_callback
        file_names, _ = self.list_files_in_folder(cloud_folder)
        logger.info(f"Starting download of {len(file_names)} files from folder: {cloud_folder}")

        for idx, file_name in enumerate(file_names, 1):
            logger.info(f"Downloading file {idx} of {len(file_names)}: {file_name}")
            date_data, time_data = get_date_time()
            callback_data = {
                "date": date_data,
                "time": time_data,
                "status": "RUNNING",
                "verbosity": "INFO",
                "message": f"Downloading file {idx} of {len(file_names)}: {file_name}",
            }
            data_string = json.dumps(callback_data)
            status_callback(data_string)

            if maintain_src_folder_structure:
                local_path = os.path.join(local_destination, file_name)
            else:
                local_path = os.path.join(local_destination, os.path.basename(file_name))
            self.download_file(file_name, local_path)

        logger.info(f"Successfully downloaded {len(file_names)} files from {cloud_folder} to {local_destination}")

    @retry_method
    @master_node_only
    def create_folder_in_bucket(self, folder):
        """Ensure a folder exists by uploading a placeholder file (optional)."""
        full_path = self.root + folder.rstrip('/') + '/.keep'
        try:
            with self.fs.open(full_path, 'wb') as f:
                f.write(b'')  # zero-byte file
        except Exception as e:
            logger.error(f"create_folder_in_bucket error: {e}")

    @retry_method
    @master_node_only
    def upload_file(self, local_file_path, cloud_file_path):
        """Upload a local file to the cloud."""
        full_path = self.root + cloud_file_path
        while full_path.find("//") != -1:
            full_path = full_path.replace("//", "/")
        try:
            self.fs.put(local_file_path, full_path)
        except Exception as e:
            logger.error(f"upload_file error: {e}")

    @retry_method
    @master_node_only
    def upload_folder(self, local_folder, cloud_subfolder):
        """Upload all files from a local folder to a cloud subfolder."""
        cloud_subfolder = cloud_subfolder.strip('/')
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder)
                cloud_object_name = f"{cloud_subfolder}/{relative_path.replace(os.path.sep, '/')}"
                self.upload_file(local_file_path, cloud_object_name)

    @retry_method
    @master_node_only
    def delete_folder(self, folder):
        """Delete a folder and its contents from the cloud storage bucket."""
        full_path = self.root + folder.rstrip('/') + '/'
        try:
            files = self.fs.find(full_path)
            for f in files:
                self.fs.rm(f)
        except Exception as e:
            logger.error(f"delete_folder error: {e}")

    @retry_method
    @master_node_only
    def delete_file(self, file_path):
        """Delete a file from the cloud storage bucket."""
        full_path = self.root + file_path
        try:
            self.fs.rm(full_path)
        except Exception as e:
            logger.error(f"delete_file error: {e}")

    @retry_method
    @master_node_only
    def move_file(self, source_path, destination_path):
        """Move a file within the cloud storage bucket."""
        src = self.root + source_path
        dst = self.root + destination_path
        try:
            self.fs.mv(src, dst)
        except Exception as e:
            logger.error(f"move_file error: {e}")

    @retry_method
    @master_node_only
    def move_folder(self, source_path, destination_path):
        """Move a folder within the cloud storage bucket."""
        src = self.root + source_path.rstrip('/') + '/'
        dst = self.root + destination_path.rstrip('/') + '/'
        try:
            files = self.fs.find(src)
            for f in files:
                rel = os.path.relpath(f, src)
                self.fs.mv(f, dst + rel)
        except Exception as e:
            logger.error(f"move_folder error: {e}")

    @retry_method
    @master_node_only
    def copy_file(self, source_object_name, destination_object_name):
        """Copy a file within the cloud storage bucket."""
        src = self.root + source_object_name
        dst = self.root + destination_object_name
        try:
            with self.fs.open(src, 'rb') as fsrc:
                with self.fs.open(dst, 'wb') as fdst:
                    fdst.write(fsrc.read())
        except Exception as e:
            logger.error(f"copy_file error: {e}")

    @retry_method
    @master_node_only
    def copy_folder(self, source_path, destination_path):
        """Copy a folder within the cloud storage bucket."""
        src = self.root + source_path.rstrip('/') + '/'
        dst = self.root + destination_path.rstrip('/') + '/'
        try:
            files = self.fs.find(src)
            for f in files:
                rel = os.path.relpath(f, src)
                with self.fs.open(f, 'rb') as fsrc:
                    with self.fs.open(dst + rel, 'wb') as fdst:
                        fdst.write(fsrc.read())
        except Exception as e:
            logger.error(f"copy_folder error: {e}")
