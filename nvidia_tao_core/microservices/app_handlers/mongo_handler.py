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
import json
import gzip
from datetime import datetime
from bson import ObjectId
import pymongo

from nvidia_tao_core.microservices.handlers.utilities import Code
from nvidia_tao_core.microservices.utils import run_system_command
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_root

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.handlers.mongo_handler import mongo_connection_string, mongo_client

from nvidia_tao_core.microservices.app_handlers.utils import get_workspace

# Configure logging
logger = logging.getLogger(__name__)


class MongoBackupHandler:
    """Handles MongoDB backup and restore operations."""

    @staticmethod
    def _serialize_document(doc):
        """Convert MongoDB document to JSON-serializable format."""
        if isinstance(doc, dict):
            return {k: MongoBackupHandler._serialize_document(v) for k, v in doc.items()}
        elif isinstance(doc, list):
            return [MongoBackupHandler._serialize_document(item) for item in doc]
        elif isinstance(doc, ObjectId):
            return {"$oid": str(doc)}
        elif isinstance(doc, datetime):
            return {"$date": doc.isoformat()}
        else:
            return doc

    @staticmethod
    def _deserialize_document(doc):
        """Convert JSON document back to MongoDB format."""
        if isinstance(doc, dict):
            if "$oid" in doc:
                return ObjectId(doc["$oid"])
            elif "$date" in doc:
                return datetime.fromisoformat(doc["$date"])
            else:
                return {k: MongoBackupHandler._deserialize_document(v) for k, v in doc.items()}
        elif isinstance(doc, list):
            return [MongoBackupHandler._deserialize_document(item) for item in doc]
        else:
            return doc

    @staticmethod
    def mongo_backup_pymongo(workspace_id, backup_file_name=None):
        """Backup MongoDB data using PyMongo (alternative to system commands).

        Args:
            workspace_id (str): ID of the workspace to backup.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation.
        """
        try:
            # Get the workspace metadata
            workspace_metadata = get_workspace(workspace_id)
            if not workspace_metadata:
                return Code(404, {}, "Workspace not found")
            
            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB backup is only supported for AWS and Azure workspaces")
            
            from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance for MongoDB backup")

            logger.info("Starting PyMongo-based MongoDB backup for workspace %s", workspace_id)

            # Create dump directory if it doesn't exist
            root = get_root()
            dump_dir = os.path.join(root, "dump", "archive")
            os.makedirs(dump_dir, exist_ok=True)

            backup_file = backup_file_name if backup_file_name else "mongodb_backup_pymongo.json.gz"
            backup_file_path = os.path.join(dump_dir, backup_file)

            # Get all database names (excluding admin, config, local)
            db_names = [name for name in mongo_client.list_database_names() 
                       if name not in ['admin', 'config', 'local']]

            backup_data = {
                "backup_timestamp": datetime.utcnow().isoformat(),
                "databases": {}
            }

            # Backup each database
            for db_name in db_names:
                database = mongo_client[db_name]
                backup_data["databases"][db_name] = {
                    "collections": {},
                    "indexes": {}
                }

                # Backup collections and their documents
                for collection_name in database.list_collection_names():
                    collection = database[collection_name]
                    
                    # Backup documents
                    documents = list(collection.find())
                    serialized_docs = [MongoBackupHandler._serialize_document(doc) for doc in documents]
                    backup_data["databases"][db_name]["collections"][collection_name] = serialized_docs
                    
                    # Backup indexes
                    indexes = list(collection.list_indexes())
                    backup_data["databases"][db_name]["indexes"][collection_name] = [
                        MongoBackupHandler._serialize_document(idx) for idx in indexes
                    ]

            # Write compressed backup file
            with gzip.open(backup_file_path, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)

            # Upload to cloud storage
            cs_instance.upload_file(backup_file_path, backup_file_path)

            logger.info("Successfully backed up MongoDB using PyMongo to cloud storage")
            return Code(200, {"message": "MongoDB backup successful (PyMongo method)"}, 
                       "MongoDB backup successful (PyMongo method)")

        except Exception as e:
            logger.error("Exception thrown in mongo_backup_pymongo: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB backup (PyMongo method)")

    @staticmethod
    def mongo_restore_pymongo(workspace_id, backup_file_name=None):
        """Restore MongoDB data using PyMongo (alternative to system commands).

        Args:
            workspace_id (str): ID of the workspace to restore.
            backup_file_name (str, optional): Name of the backup file.

        Returns:
            Response: A response indicating the outcome of the operation.
        """
        try:
            # Get the workspace metadata
            workspace_metadata = get_workspace(workspace_id)
            if not workspace_metadata:
                return Code(404, {}, "Workspace not found")

            cloud_type = workspace_metadata.get("cloud_type")
            if cloud_type not in ["aws", "azure"]:
                return Code(400, {}, "MongoDB restore is only supported for AWS and Azure workspaces")
            
            from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
            cs_instance, _ = create_cs_instance(workspace_metadata)
            if not cs_instance:
                return Code(404, {}, "Unable to create cloud storage instance for MongoDB restore")

            logger.info("Starting PyMongo-based MongoDB restore for workspace %s", workspace_id)

            root = get_root()
            dump_dir = os.path.join(root, "dump", "archive")
            backup_file = backup_file_name if backup_file_name else "mongodb_backup_pymongo.json.gz"
            backup_file_path = os.path.join(dump_dir, backup_file)

            # Download backup file from cloud storage
            cs_instance.download_file(backup_file_path, backup_file_path)
            logger.info("Downloaded backup file to %s", backup_file_path)

            # Read and decompress backup file
            with gzip.open(backup_file_path, 'rt', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Restore each database
            for db_name, db_data in backup_data["databases"].items():
                database = mongo_client[db_name]

                # Restore collections
                for collection_name, documents in db_data["collections"].items():
                    collection = database[collection_name]
                    
                    # Clear existing data (optional - you might want to make this configurable)
                    collection.drop()
                    
                    if documents:  # Only insert if there are documents
                        # Deserialize documents
                        deserialized_docs = [MongoBackupHandler._deserialize_document(doc) for doc in documents]
                        collection.insert_many(deserialized_docs)

                # Restore indexes (skip the default _id index)
                if "indexes" in db_data:
                    for collection_name, indexes in db_data["indexes"].items():
                        collection = database[collection_name]
                        for index_info in indexes:
                            try:
                                if index_info.get("name") != "_id_":  # Skip default index
                                    deserialized_index = MongoBackupHandler._deserialize_document(index_info)
                                    # Extract index specification
                                    if "key" in deserialized_index:
                                        index_spec = list(deserialized_index["key"].items())
                                        index_options = {k: v for k, v in deserialized_index.items() 
                                                       if k not in ["key", "v", "ns"]}
                                        collection.create_index(index_spec, **index_options)
                            except Exception as idx_error:
                                logger.warning("Failed to restore index %s: %s", 
                                             index_info.get("name", "unknown"), str(idx_error))

            # Clean up backup file
            os.remove(backup_file_path)
            
            logger.info("Restored MongoDB from backup file using PyMongo")
            return Code(200, {"message": "MongoDB restore successful (PyMongo method)"}, 
                       "MongoDB restore successful (PyMongo method)")

        except Exception as e:
            logger.error("Exception thrown in mongo_restore_pymongo: %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Error in MongoDB restore (PyMongo method)")

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
            from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
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
            from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
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
