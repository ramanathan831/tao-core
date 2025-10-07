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

"""MongoDB backup and restore handler module"""
import os
import logging
import traceback

from nvidia_tao_core.microservices.handlers.utilities import Code
from nvidia_tao_core.microservices.utils import run_system_command
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_root

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.handlers.mongo_handler import mongo_connection_string

from nvidia_tao_core.microservices.app_handlers.utils import get_workspace

# Configure logging
logger = logging.getLogger(__name__)


class MongoBackupHandler:
    """Handles MongoDB backup and restore operations."""

    @staticmethod
    def mongo_backup(workspace_id, backup_file_name=None):
        """Backup MongoDB data for a specific workspace.

        Args:
            workspace_id (str): ID of the workspace to backup.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        try:
            # Get the workspace metadata
            workspace_metadata = get_workspace(workspace_id)
            if not workspace_metadata:
                return Code(404, {}, "Workspace not found")
            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB backup is only supported for AWS and Azure workspaces")
            from nvidia_tao_core.microservices.handlers.cloud_handlers.cloud_storage import create_cs_instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance for MongoDB backup")

            logger.info("Starting MongoDB backup for workspace %s", workspace_id)

            # Create dump directory if it doesn't exist
            root = get_root()
            dump_dir = os.path.join(root, "dump", "archive")
            os.makedirs(dump_dir, exist_ok=True)

            backup_file = backup_file_name if backup_file_name else "mongodb_backup.gz"
            backup_file = os.path.join(dump_dir, backup_file)
            backup_command = f'mongodump --uri="{mongo_connection_string}" --archive="{backup_file}" --gzip'
            run_system_command(backup_command)

            cs_instance.upload_file(backup_file, backup_file)

            logger.info("Successfully backed up MongoDB to S3")
            return Code(200, {"message": "MongoDB backup successful"}, "MongoDB backup successful")

        except Exception as e:
            logger.error("Exception thrown in mongo_backup is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB backup")

    @staticmethod
    def mongo_restore(workspace_id, backup_file_name=None):
        """Restore MongoDB data for a specific workspace.

        Args:
            workspace_id (str): ID of the workspace to restore.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        try:
            # Get the workspace metadata
            workspace_metadata = get_workspace(workspace_id)
            if not workspace_metadata:
                return Code(404, {}, "Workspace not found")

            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB restore is only supported for AWS and Azure workspaces")
            from nvidia_tao_core.microservices.handlers.cloud_handlers.cloud_storage import create_cs_instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance for MongoDB restore")

            root = get_root()
            dump_dir = os.path.join(root, "dump", "archive")
            backup_file = backup_file_name if backup_file_name else "mongodb_backup.gz"
            backup_file = os.path.join(dump_dir, backup_file)
            cs_instance.download_file(backup_file, backup_file)
            logger.info(f"Downloaded backup file to {backup_file}")
            restore_command = f'mongorestore --uri="{mongo_connection_string}" --archive="{backup_file}" --gzip'
            run_system_command(restore_command)
            logger.info("Restored DB from backup file")
            os.remove(backup_file)
            return Code(200, {"message": "MongoDB restore successful"}, "MongoDB restore successful")

        except Exception as e:
            logger.error("Exception thrown in mongo_restore is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB restore")
