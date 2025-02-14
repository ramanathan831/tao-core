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


import sys
import json
import glob
import importlib
import os
import threading
import traceback
import yaml
from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.cloud_handlers.utils import download_files_from_spec, get_results_cloud_data, monitor_and_upload
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points

module = entry_points[0].module_name.split('.')[0]
entrypoint = importlib.import_module(entrypoint_paths[module])


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
        cloud_storage = None
        exit_event = None
        upload_thread = None
        status_logger = None

        try:
            # Extract job configuration
            env_vars = {
                "CLOUD_BASED": job.get('hosted_service_interaction', ""),
                "NVCF_HELM": job.get('nvcf_helm', ""),
                "TELEMETRY_OPT_OUT": job.get('telemetry_opt_out', "no"),
                "TAO_USER_KEY": job.get('ngc_key', ""),
                "TAO_ADMIN_KEY": job.get('tao_api_admin_key', ""),
                "TAO_API_SERVER": job.get('tao_api_base_url', ""),
                "TAO_LOGGING_SERVER_URL": job.get('tao_api_status_callback_url', ""),
                "AUTOML_EXPERIMENT_NUMBER": job.get('automl_experiment_number', ""),
                "JOB_ID": job["job_id"]
            }

            # Update environment variables
            os.environ.update(env_vars)
            docker_env_vars = job.get('docker_env_vars')
            if docker_env_vars:
                os.environ.update(docker_env_vars)

            # Setup cloud storage and specs
            cloud_storage, specs = get_results_cloud_data(
                job.get("cloud_metadata"),
                job["specs"],
                f'/results/{job["job_id"]}'
            )

            # Create results directory and download files
            os.makedirs(specs["results_dir"], exist_ok=True)
            download_files_from_spec(
                cloud_data=job.get("cloud_metadata"),
                data=specs,
                job_id=job["job_id"],
                network_arch=job["neural_network_name"],
                ngc_key=job.get("ngc_key"),
                tao_api_ui_cookie=job.get('tao_api_ui_cookie', ""),
                use_ngc_staging=job.get('use_ngc_staging', "False")
            )

            # Save spec file
            spec_path = os.path.join(specs["results_dir"], "spec.yaml")
            with open(spec_path, 'w+', encoding='utf-8') as yaml_file:
                yaml.dump(specs, yaml_file, default_flow_style=False)

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
                status_file = ContainerJobHandler.get_status_file(specs["results_dir"], job["action_name"])

                try:
                    # Initialize status logger
                    status_logger = status_logging.StatusLogger(
                        filename=status_file,
                        is_master=True,
                        verbosity=1,
                        append=True
                    )
                    status_logging.set_status_logger(status_logger)

                    # Launch entrypoint
                    _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                    is_completed = entrypoint.launch(args, "", actions, network=job["neural_network_name"])

                except Exception:
                    print("Traceback", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    ContainerJobHandler._handle_failure(job, status_logger, status_file)
                finally:
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

            return job["job_id"]

        except Exception:
            print("Traceback", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if status_logger:
                status_logging.get_status_logger().write(
                    message=f"{job['action_name']} action couldn't be launched for {job['neural_network_name']}",
                    status_level=status_logging.Status.FAILURE
                )
            ContainerJobHandler._cleanup(exit_event, upload_thread)
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
                print("Failed to create status logger", file=sys.stderr)
        status_logging.get_status_logger().write(
            message=f"{job['action_name']} action failed for {job['neural_network_name']}",
            status_level=status_logging.Status.FAILURE
        )

    @staticmethod
    def _cleanup(exit_event=None, upload_thread=None, job=None, is_completed=None, status_logger=None, status_file=None):
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
                    print("Failed to create status logger", file=sys.stderr)

            status_logging.get_status_logger().write(
                message=f"{job['action_name']} action {result} for {job['neural_network_name']}",
                status_level=status
            )

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
    def get_current_job_status(specs):
        """Finds 'status.json' under specs['results_dir'] and returns the last entry's status."""
        results_dir = specs.get("results_dir")
        if "://" in results_dir:
            bucket_name = results_dir.split("//")[1].split("/")[0]
            results_dir = results_dir[results_dir.find(bucket_name) + len(bucket_name):]

        if not results_dir or not os.path.isdir(results_dir):
            raise ValueError("Invalid or missing 'results_dir' in specs.")

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
