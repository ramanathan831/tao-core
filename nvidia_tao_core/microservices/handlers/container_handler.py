# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/NVIDIA/NeMo
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

"""Job queue handling."""


import json
import glob
import importlib
import os
import threading
import traceback
import yaml
import logging

from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.api_utils.entrypoint_mimicker import vlm_entrypoint
from nvidia_tao_core.cloud_handlers.utils import (
    download_files_from_spec,
    get_results_cloud_data,
    monitor_and_upload,
    cleanup_cuda_contexts,
)
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points
from nvidia_tao_core.microservices.utils import safe_load_file, safe_dump_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

module = entry_points[0].module_name.split('.')[0] if entry_points else None
entrypoint = importlib.import_module(entrypoint_paths[module]) if module else None


class ContainerJobHandler:
    """Handler for processing jobs in a containerized environment."""

    @staticmethod
    def entrypoint_wrapper(job):
        """Starts job asynchronously and handles cleanup after completion.

        Args:
            job (dict): Job configuration containing action, neural network and other metadata

        Returns:
            str: Job ID if launch successful, None otherwise
        """
        try:
            docker_env_vars = job.get('docker_env_vars', {})
            if docker_env_vars:
                os.environ.update(docker_env_vars)

            def async_setup_and_run():
                cloud_storage = None
                exit_event = None
                upload_thread = None
                status_logger = None

                try:
                    # Setup cloud storage and specs
                    cloud_storage, specs = get_results_cloud_data(
                        job.get("cloud_metadata"),
                        job["specs"],
                        f'/results/{job["job_id"]}'
                    )

                    # Create results directory and download files
                    os.makedirs(specs["results_dir"], exist_ok=True)
                    reprocess_files = []
                    download_files_from_spec(
                        cloud_data=job.get("cloud_metadata"),
                        data=specs,
                        job_id=job["job_id"],
                        network_arch=job["neural_network_name"],
                        ngc_key=docker_env_vars.get("TAO_USER_KEY"),
                        tao_api_ui_cookie=docker_env_vars.get('TAO_API_UI_COOKIE', ""),
                        use_ngc_staging=docker_env_vars.get('USE_NGC_STAGING', "False"),
                        reprocess_files=reprocess_files
                    )

                    # Save spec file
                    spec_path = os.path.join(specs["results_dir"], "spec.yaml")
                    with open(spec_path, 'w+', encoding='utf-8') as yaml_file:
                        yaml.dump(specs, yaml_file, default_flow_style=False)

                    if docker_env_vars.get("RECURSIVE_DATASET_FILE_DOWNLOAD", "False") == "True":
                        logger.info("reprocess_files: %s", reprocess_files)
                        if reprocess_files:
                            for file_name in reprocess_files:
                                file_type = file_name.split(".")[-1]
                                reprocess_file_data = safe_load_file(file_name, file_type=file_type)
                                if reprocess_file_data:
                                    download_files_from_spec(
                                        cloud_data=job.get("cloud_metadata"),
                                        data=reprocess_file_data,
                                        job_id=job["job_id"],
                                        network_arch=job["neural_network_name"],
                                        ngc_key=docker_env_vars.get("TAO_USER_KEY"),
                                        tao_api_ui_cookie=docker_env_vars.get('TAO_API_UI_COOKIE', ""),
                                        use_ngc_staging=docker_env_vars.get('USE_NGC_STAGING', "False"),
                                    )
                                    if reprocess_file_data:
                                        safe_dump_file(file_name, reprocess_file_data, file_type=file_type)

                    # Start cloud upload monitoring if needed
                    if cloud_storage:
                        exit_event = threading.Event()
                        upload_thread = threading.Thread(
                            target=monitor_and_upload,
                            args=(specs["results_dir"], cloud_storage, exit_event),
                            daemon=True
                        )
                        upload_thread.start()

                    # Prepare entrypoint arguments
                    args = {
                        "subtask": job["action_name"],
                        "experiment_spec_file": spec_path,
                        "results_dir": specs["results_dir"]
                    }

                    def run_entrypoint():
                        nonlocal status_logger
                        is_completed = False
                        status_file = None

                        def initialize_status_logger(status_file):
                            """Initialize or get existing status logger."""
                            nonlocal status_logger
                            if not status_logger:
                                status_logger = status_logging.StatusLogger(
                                    filename=status_file,
                                    is_master=True,
                                    verbosity=1,
                                    append=True
                                )
                                status_logging.set_status_logger(status_logger)
                            return status_logger

                        try:
                            # Launch entrypoint
                            if entrypoint:
                                try:
                                    _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                                    entrypoint.launch(args, "", actions, network=job["neural_network_name"])
                                    is_completed = True
                                except SystemExit as e:
                                    is_completed = e.code == 0
                            else:
                                is_completed = vlm_entrypoint.vlm_launch(
                                    job["neural_network_name"],
                                    job["action_name"],
                                    specs
                                )

                        except Exception:
                            logger.error("Traceback")
                            logger.error(traceback.format_exc())
                            status_file = ContainerJobHandler.get_status_file(specs["results_dir"], job["action_name"])
                            status_logger = initialize_status_logger(status_file)
                            ContainerJobHandler._handle_failure(job, status_logger, status_file)
                        finally:
                            status_file = status_file or ContainerJobHandler.get_status_file(
                                specs["results_dir"],
                                job["action_name"]
                            )
                            status_logger = initialize_status_logger(status_file)
                            ContainerJobHandler._cleanup(
                                exit_event,
                                upload_thread,
                                job,
                                is_completed,
                                status_logger,
                                status_file
                            )

                    # Launch job asynchronously
                    entrypoint_thread = threading.Thread(target=run_entrypoint, daemon=True)
                    entrypoint_thread.start()

                except Exception:
                    logger.error("Traceback")
                    logger.error(traceback.format_exc())
                    if status_logger:
                        status_logging.get_status_logger().write(
                            message=(
                                f"{job['action_name']} action couldn't be launched "
                                f"for {job['neural_network_name']}"
                            ),
                            status_level=status_logging.Status.FAILURE
                        )
                    ContainerJobHandler._cleanup(exit_event, upload_thread)

            # Launch the async setup and execution
            setup_thread = threading.Thread(target=async_setup_and_run, daemon=True)
            setup_thread.start()

            return job["job_id"]

        except Exception:
            logger.error("Traceback")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _handle_failure(job, status_logger, status_file):
        """Handle failure cases by creating status logger if needed and logging failure."""
        if not status_logger:
            try:
                status_logger = status_logging.StatusLogger(
                    filename=status_file,
                    is_master=True,
                    verbosity=1,
                    append=True
                )
                status_logging.set_status_logger(status_logger)
            except Exception:
                logger.error("Failed to create status logger")
        status_logging.get_status_logger().write(
            message=f"{job['action_name']} action failed for {job['neural_network_name']}",
            status_level=status_logging.Status.FAILURE
        )

    @staticmethod
    def _cleanup(
        exit_event=None,
        upload_thread=None,
        job=None,
        is_completed=None,
        status_logger=None,
        status_file=None
    ):
        """Clean up resources and log final status."""
        if exit_event:
            exit_event.set()
        if upload_thread:
            upload_thread.join()

        if job and is_completed is not None:
            status = status_logging.Status.SUCCESS if is_completed else status_logging.Status.FAILURE
            result = "completed successfully" if is_completed else "failed"

            if not status_logger:
                try:
                    status_logger = status_logging.StatusLogger(
                        filename=status_file,
                        is_master=True,
                        verbosity=1,
                        append=True
                    )
                    status_logging.set_status_logger(status_logger)
                except Exception:
                    logger.error("Failed to create status logger")

            status_logging.get_status_logger().write(
                message=f"{job['action_name']} action {result} for {job['neural_network_name']}",
                status_level=status
            )
        # Clean up any stale CUDA contexts
        cleanup_cuda_contexts()

    @staticmethod
    def get_status_file(results_dir, action_name=""):
        """Get the path to the status file.

        Args:
            results_dir (str): Directory containing results and status files
            action_name (str, optional): Name of action for creating new status file path. Defaults to "".

        Returns:
            str: Path to existing status.json file if found, otherwise constructs new path
        """
        status_files = glob.glob(os.path.join(results_dir, "**", "status.json"), recursive=True)
        if not status_files:
            return os.path.join(results_dir, action_name, "status.json")
        return status_files[0]

    @staticmethod
    def get_current_job_status(results_dir):
        """Finds 'status.json' under specs['results_dir'] and returns the last entry's status."""
        if "://" in results_dir:
            bucket_name = results_dir.split("//")[1].split("/")[0]
            results_dir = results_dir[results_dir.find(bucket_name) + len(bucket_name):]

        if not results_dir:
            cleanup_cuda_contexts()
            raise ValueError("Empty 'results_dir' in specs.")
        if not os.path.isdir(results_dir):
            logger.error("results_dir directory %s does not exist", results_dir)
            return "Pending"

        file_path = ContainerJobHandler.get_status_file(results_dir)
        last_status = None

        if not os.path.exists(file_path):
            return "Pending"

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    last_status = data.get("status")
                except json.JSONDecodeError:
                    continue

        return {
            "STARTED": "Running",
            "RUNNING": "Running",
            "SUCCESS": "Done",
            "FAILURE": "Error",
        }.get(last_status, "Pending")
