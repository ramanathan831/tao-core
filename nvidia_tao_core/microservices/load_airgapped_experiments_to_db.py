#!/usr/bin/env python3

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

"""Load air-gapped experiment metadata from JSON and import to database"""
import argparse
import os
import logging
import sys
import yaml
import glob
import tempfile

from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
from nvidia_tao_core.microservices.utils import safe_load_file
from nvidia_tao_core.cloud_handlers.utils import initialize_cloud_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

base_exp_uuid = "00000000-0000-0000-0000-000000000000"


class AirgappedExperimentLoader:
    """Loader for air-gapped experiment metadata"""

    def __init__(self, json_file_path: str = None, dry_run: bool = False, models_base_dir: str = None,
                 use_cloud_storage: bool = False, cloud_config: dict = None):
        """Initialize the loader

        Args:
            json_file_path (str, optional): Path to the JSON file containing experiment metadata.
                                          If use_cloud_storage=True, defaults to "index.json" under
                                          LOCAL_MODEL_REGISTRY. If use_cloud_storage=False, this parameter is required.
            dry_run (bool, optional): If True, only validate data without writing to database. Defaults to False.
            models_base_dir (str, optional): Base directory for searching model files. Defaults to JSON file directory.
            use_cloud_storage (bool, optional): If True, download JSON file and experiment.yaml from cloud
                                          storage. Defaults to False.
            cloud_config (dict, optional): Cloud storage configuration. Required if use_cloud_storage=True.
        """
        self.dry_run = dry_run
        self.mongo_handler = MongoHandler("tao", "experiments")
        self.use_cloud_storage = use_cloud_storage
        self.local_json_file = None

        if use_cloud_storage:
            if not cloud_config:
                raise ValueError("cloud_config is required when use_cloud_storage=True")

            # Use fixed path "index.json" for cloud storage
            self.json_file_path = json_file_path or "index.json"

            # Initialize cloud storage
            self.cloud_storage = initialize_cloud_storage(
                cloud_type=cloud_config.get("cloud_type", "seaweedfs"),
                bucket_name=cloud_config.get("bucket_name", "tao-storage"),
                region=cloud_config.get("region"),
                access_key=cloud_config.get("access_key"),
                secret_key=cloud_config.get("secret_key"),
                endpoint_url=cloud_config.get("endpoint_url")
            )
            logger.info("Initialized cloud storage for downloading JSON and experiment.yaml files")

            # Download JSON file from cloud storage
            self.local_json_file = self._download_json_file_from_cloud()
            if not self.local_json_file:
                raise FileNotFoundError(f"Failed to download JSON file from cloud storage: {self.json_file_path}")
        else:
            if not json_file_path:
                raise ValueError("json_file_path is required when use_cloud_storage=False")

            self.json_file_path = json_file_path
            self.cloud_storage = None
            self.local_json_file = json_file_path
            # Base directory for searching spec files (local filesystem)
            self.base_dir = models_base_dir if models_base_dir else os.path.dirname(os.path.abspath(json_file_path))
            logger.info("Using base directory for model file search: %s", self.base_dir)

    def _download_json_file_from_cloud(self):
        """Download JSON file from cloud storage under LOCAL_MODEL_REGISTRY folder

        Returns:
            str: Path to local temporary JSON file, or None if download failed
        """
        try:
            # Construct cloud path using LOCAL_MODEL_REGISTRY environment variable
            local_model_registry = os.getenv('LOCAL_MODEL_REGISTRY')
            if not local_model_registry:
                raise ValueError("LOCAL_MODEL_REGISTRY environment variable is not set")

            # Ensure json_file_path is relative to LOCAL_MODEL_REGISTRY
            if self.json_file_path.startswith('/'):
                # Remove leading slash if present
                json_file_path = self.json_file_path.lstrip('/')
            else:
                json_file_path = self.json_file_path

            cloud_json_path = f"{local_model_registry}/{json_file_path}"
            logger.info("Downloading JSON file from cloud storage: %s", cloud_json_path)

            # Check if file exists in cloud storage
            if not self.cloud_storage.is_file(cloud_json_path):
                logger.error("JSON file not found in cloud storage: %s", cloud_json_path)
                return None

            # Create temporary file for download
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Download the file
                self.cloud_storage.download_file(cloud_json_path, temp_path)
                logger.info("Successfully downloaded JSON file to: %s", temp_path)
                return temp_path

            except Exception as e:
                logger.error("Failed to download JSON file from cloud storage: %s", e)
                # Clean up temporary file if download failed
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                return None

        except Exception as e:
            logger.error("Error setting up JSON file download from cloud storage: %s", e)
            return None

    def load_experiments_from_json(self):
        """Load experiments from JSON file"""
        json_file_to_use = self.local_json_file

        if not os.path.isfile(json_file_to_use):
            raise FileNotFoundError(f"JSON file not found: {json_file_to_use}")

        logger.info("Loading experiments from JSON file: %s", json_file_to_use)

        try:
            experiments = safe_load_file(json_file_to_use)
            if isinstance(experiments, list):
                # Convert list to dictionary with id as key for consistency
                experiments_dict = {exp.get("id"): exp for exp in experiments if exp.get("id")}
                logger.info("Loaded %d experiments from JSON file", len(experiments_dict))
                return experiments_dict
            if isinstance(experiments, dict):
                logger.info("Loaded %d experiments from JSON file", len(experiments))
                return experiments
            raise ValueError("Invalid JSON format: expected list or dictionary")
        except Exception as e:
            logger.error("Failed to load experiments from JSON: %s", e)
            raise

    def find_and_read_experiment_yaml(self, experiment):
        """Find and read experiment.yaml file for a given experiment

        Args:
            experiment (dict): Experiment metadata containing ngc_path

        Returns:
            dict or None: Parsed YAML content if file found and valid, None otherwise
        """
        ngc_path = experiment.get("ngc_path")
        if not ngc_path:
            logger.warning("No ngc_path found for experiment: %s", experiment.get("name", "unknown"))
            return None

        # Split ngc_path: "nvstaging/tao/ocrnet:trainable_v2.0" -> "nvstaging/tao/ocrnet", "trainable_v2.0"
        if ":" not in ngc_path:
            logger.warning("Invalid ngc_path format (missing version): %s", ngc_path)
            return None

        path_part, version = ngc_path.split(":", 1)

        if self.use_cloud_storage:
            return self._download_experiment_yaml_from_cloud(experiment, path_part, version)
        return self._read_experiment_yaml_from_local(experiment, path_part, version)

    def _download_experiment_yaml_from_cloud(self, experiment, path_part, version):
        """Download experiment.yaml from cloud storage"""
        try:
            # Construct cloud path: /data/nvstaging/tao/ocrnet/trainable_v2.0/
            cloud_dir = f"{os.getenv('LOCAL_MODEL_REGISTRY')}/{path_part}/{version}/"
            logger.debug("Searching for model directories in cloud path: %s", cloud_dir)

            # List directories to find the model subdirectory
            files, _ = self.cloud_storage.list_files_in_folder(cloud_dir)
            logger.debug("Found files: %s", files)

            # Look for experiment.yaml in each subdirectory
            for file in files:
                if not file.endswith("experiment.yaml"):
                    continue

                if self.cloud_storage.is_file(file):
                    # Download to temporary file
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as temp_file:
                        temp_path = temp_file.name

                    try:
                        self.cloud_storage.download_file(file, temp_path)
                        logger.debug("Downloaded experiment.yaml to temporary file: %s", temp_path)

                        # Read and parse YAML content
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            yaml_content = yaml.safe_load(f)
                            logger.debug("Successfully loaded YAML content from cloud storage")
                            return yaml_content

                    except Exception as e:
                        logger.error("Failed to download/parse YAML file %s: %s", file, e)
                        return None

                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logger.error("Failed to clean up temporary file %s: %s", temp_path, e)
                            pass

            logger.warning("No experiment.yaml file found in cloud storage for experiment: %s (searched in %s)",
                           experiment.get("name", "unknown"), cloud_dir)
            return None

        except Exception as e:
            logger.error("Error accessing cloud storage for experiment %s: %s",
                         experiment.get("name", "unknown"), e)
            return None

    def _read_experiment_yaml_from_local(self, experiment, path_part, version):
        """Read experiment.yaml from local filesystem"""
        # Construct search pattern: base_dir/ngc_path_part/version/*/experiment.yaml
        search_pattern = os.path.join(self.base_dir, path_part, version, "*", "experiment.yaml")
        logger.debug("Searching for spec file with pattern: %s", search_pattern)

        matches = glob.glob(search_pattern)
        if matches:
            yaml_file = matches[0]  # Use first match
            logger.info("Found experiment.yaml for %s: %s", experiment.get("name", "unknown"), yaml_file)

            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                    logger.debug("Successfully loaded YAML content from %s", yaml_file)
                    return yaml_content
            except Exception as e:
                logger.error("Failed to read YAML file %s: %s", yaml_file, e)
                return None

        logger.warning("No experiment.yaml file found for experiment: %s (pattern: %s)",
                       experiment.get("name", "unknown"), search_pattern)
        return None

    def validate_experiment_data(self, experiments):
        """Validate experiment data before database import"""
        logger.info("Validating experiment data...")

        required_fields = [
            "id", "name", "network_arch", "ngc_path", "actions",
            "accepted_dataset_intents", "dataset_type"
        ]

        valid_experiments = {}
        invalid_count = 0

        for exp_id, experiment in experiments.items():
            try:
                # Check required fields
                missing_fields = [field for field in required_fields if field not in experiment]
                if missing_fields:
                    logger.warning("Experiment %s missing required fields: %s", exp_id, missing_fields)
                    invalid_count += 1
                    continue

                # Validate experiment ID matches
                if experiment["id"] != exp_id:
                    logger.warning("Experiment ID mismatch: key=%s, id=%s", exp_id, experiment["id"])
                    invalid_count += 1
                    continue

                # Ensure base_experiment_metadata exists
                if "base_experiment_metadata" not in experiment:
                    logger.warning("Experiment %s missing base_experiment_metadata", exp_id)
                    experiment["base_experiment_metadata"] = {}

                # Handle spec file reading if spec_file_present is true
                base_meta = experiment["base_experiment_metadata"]
                if base_meta.get("spec_file_present"):
                    logger.info("Reading experiment.yaml for experiment: %s", exp_id)
                    yaml_content = self.find_and_read_experiment_yaml(experiment)
                    if yaml_content:
                        base_meta["specs"] = yaml_content
                        logger.info("Successfully loaded specs for experiment: %s", exp_id)
                    else:
                        logger.warning(
                            "Failed to load specs for experiment %s despite spec_file_present=true, "
                            "setting spec_file_present=false", exp_id
                        )
                        base_meta["spec_file_present"] = False
                        base_meta["specs"] = {}
                else:
                    # Ensure specs field exists even when spec_file_present is false
                    if "specs" not in base_meta:
                        base_meta["specs"] = {}

                valid_experiments[exp_id] = experiment

            except Exception as e:
                logger.error("Error validating experiment %s: %s", exp_id, e)
                invalid_count += 1
                continue

        logger.info("Validation complete: %d valid, %d invalid experiments",
                    len(valid_experiments), invalid_count)
        return valid_experiments

    def import_to_database(self, experiments):
        """Import experiments to database"""
        if self.dry_run:
            logger.info("DRY RUN: Would import %d experiments to database", len(experiments))
            return

        logger.info("Importing %d experiments to database...", len(experiments))

        success_count = 0
        error_count = 0

        for exp_id, experiment in experiments.items():
            try:
                self.mongo_handler.upsert({'id': exp_id}, experiment)
                success_count += 1
                logger.debug("Successfully imported experiment: %s", exp_id)
            except Exception as e:
                logger.error("Failed to import experiment %s: %s", exp_id, e)
                error_count += 1

        logger.info("Import complete: %d successful, %d failed", success_count, error_count)

    def load_and_import(self):
        """Load experiments from JSON and import to database"""
        try:
            # Load experiments from JSON
            experiments = self.load_experiments_from_json()

            # Validate data
            valid_experiments = self.validate_experiment_data(experiments)

            if not valid_experiments:
                logger.error("No valid experiments found to import")
                return False

            # Import to database
            self.import_to_database(valid_experiments)

            logger.info("Air-gapped experiment import completed successfully")
            return True

        except Exception as e:
            logger.error("Failed to load and import experiments: %s", e)
            return False
        finally:
            # Clean up temporary files
            self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        if self.use_cloud_storage and self.local_json_file and os.path.exists(self.local_json_file):
            try:
                os.unlink(self.local_json_file)
                logger.debug("Cleaned up temporary JSON file: %s", self.local_json_file)
            except Exception as e:
                logger.error("Failed to clean up temporary JSON file %s: %s", self.local_json_file, e)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load air-gapped experiment metadata from JSON to database")
    parser.add_argument(
        "json_file",
        nargs='?',
        help="Path to JSON file containing experiment metadata. "
             "If --use-cloud-storage is specified, defaults to 'index.json' under LOCAL_MODEL_REGISTRY folder. "
             "Otherwise, this parameter is required for local file operations."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without writing to database"
    )
    parser.add_argument(
        "--models-base-dir",
        help="Base directory for searching model files (defaults to JSON file directory)"
    )
    parser.add_argument(
        "--use-cloud-storage",
        action="store_true",
        help="Download experiment.yaml files from cloud storage instead of local filesystem"
    )
    parser.add_argument(
        "--cloud-type",
        default="seaweedfs",
        help="Cloud storage type (default: seaweedfs)"
    )
    parser.add_argument(
        "--bucket-name",
        default="tao-storage",
        help="Cloud storage bucket name (default: tao-storage)"
    )
    parser.add_argument(
        "--endpoint-url",
        help="Cloud storage endpoint URL"
    )
    parser.add_argument(
        "--access-key",
        help="Cloud storage access key"
    )
    parser.add_argument(
        "--secret-key",
        help="Cloud storage secret key"
    )
    parser.add_argument(
        "--region",
        help="Cloud storage region"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows debug statements)"
    )

    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        # Set debug level only for this module's logger, not globally
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled - debug statements will be shown")

    # Prepare cloud configuration if using cloud storage
    cloud_config = None
    if args.use_cloud_storage:
        # Use command line args or fall back to environment variables
        endpoint_url = args.endpoint_url
        access_key = args.access_key
        secret_key = args.secret_key
        bucket_name = args.bucket_name

        cloud_config = {
            "cloud_type": args.cloud_type,
            "bucket_name": bucket_name,
            "endpoint_url": endpoint_url,
            "access_key": access_key,
            "secret_key": secret_key,
            "region": args.region
        }

        # Validate required cloud storage parameters
        if not all([endpoint_url, access_key, secret_key]):
            logger.error(
                "When using --use-cloud-storage, you must provide endpoint-url, access-key, and secret-key "
                "via command line args or environment variables "
                "(SEAWEEDFS_S3_ENDPOINT, SEAWEEDFS_ACCESS_KEY, SEAWEEDFS_SECRET_KEY)"
            )
            sys.exit(1)

        logger.info("Using cloud storage configuration: endpoint=%s, bucket=%s", endpoint_url, bucket_name)

    # Validate json_file argument based on use_cloud_storage
    if not args.use_cloud_storage and not args.json_file:
        logger.error("json_file argument is required when not using cloud storage")
        sys.exit(1)

    # Initialize loader
    loader = AirgappedExperimentLoader(
        args.json_file,
        args.dry_run,
        args.models_base_dir,
        args.use_cloud_storage,
        cloud_config
    )

    # Load and import
    success = loader.load_and_import()

    if success:
        logger.info("Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
