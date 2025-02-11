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
import importlib
import os
import threading
import time
import traceback
import yaml
from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.cloud_handlers.utils import download_files_from_spec, get_results_cloud_data, monitor_and_upload
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points

module = entry_points[0].module_name.split('.')[0]
entrypoint = importlib.import_module(entrypoint_paths[module])

# Function to process jobs from the queue
def entrypoint_wrapper(job):
    """Scans for new jobs and pushes them to running state"""
    cloud_storage = None
    is_completed = None
    exit_event = None
    upload_thread = None

    try:
        telemetry_opt_out = job["data"].get('telemetry_opt_out', "no")
        nvcf_helm = job["data"].get('nvcf_helm', "")
        use_ngc_staging = job["data"].get('use_ngc_staging', "False")
        tao_api_ui_cookie = job["data"].get('tao_api_ui_cookie', "")
        tao_api_user_key = job["data"].get('ngc_key', "")
        tao_api_admin_key = job["data"].get('tao_api_admin_key', "")
        tao_api_base_url = job["data"].get('tao_api_base_url', "")
        tao_api_status_callback_url = job["data"].get('tao_api_status_callback_url', "")
        automl_experiment_number = job["data"].get('automl_experiment_number', "")
        hosted_service_interaction = job["data"].get('hosted_service_interaction', "")
        docker_env_vars = job["data"].get('docker_env_vars', {})

        if docker_env_vars:
            os.environ.update(docker_env_vars)

        os.environ["CLOUD_BASED"] = hosted_service_interaction
        if nvcf_helm:
            os.environ["NVCF_HELM"] = nvcf_helm
        os.environ["TELEMETRY_OPT_OUT"] = telemetry_opt_out
        os.environ["TAO_USER_KEY"] = tao_api_user_key
        os.environ["TAO_ADMIN_KEY"] = tao_api_admin_key
        os.environ["TAO_API_SERVER"] = tao_api_base_url
        os.environ["TAO_LOGGING_SERVER_URL"] = tao_api_status_callback_url
        os.environ["AUTOML_EXPERIMENT_NUMBER"] = automl_experiment_number
        os.environ["JOB_ID"] = job["job_id"]

        # Obtaining the CloudStroage instance to save the results
        cloud_storage, specs = get_results_cloud_data(job["data"].get("cloud_metadata"), job["data"]["specs"], f'/results/{job["job_id"]}')

        # Creating the job directory
        os.makedirs(specs["results_dir"], exist_ok=True)

        download_files_from_spec(cloud_data=job["data"].get("cloud_metadata"),
                                    data=specs,
                                    job_id=job["job_id"],
                                    network_arch=job["neural_network_name"],
                                    ngc_key=job["data"].get("ngc_key"),
                                    tao_api_ui_cookie=tao_api_ui_cookie,
                                    use_ngc_staging=use_ngc_staging,
                                    )

        # Creating the Spec
        with open(f'{specs["results_dir"]}/spec.yaml', 'w+', encoding='utf-8') as yaml_file:
            yaml.dump(specs, yaml_file, default_flow_style=False)

        # Starting the thread to update the results to the cloud
        if cloud_storage:
            exit_event = threading.Event()
            upload_thread = threading.Thread(target=monitor_and_upload, args=(specs["results_dir"], cloud_storage, exit_event), daemon=True)
            upload_thread.start()

        # Creating the request object and launching the action
        args = {
            "subtask": job["action"],
            "experiment_spec_file": f'{specs["results_dir"]}/spec.yaml',
            "results_dir": specs["results_dir"],
        }

        _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
        is_completed = entrypoint.launch(args, "", actions, network=job["neural_network_name"])

    except Exception:
        print("Traceback", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        status_logging.get_status_logger().write(
            message=f"{job['action']} action couldn't be launched for {job['neural_network_name']}",
            status_level=status_logging.Status.FAILURE
        )
    finally:
        # Stopping the threads
        if cloud_storage:
            if exit_event is not None:
                exit_event.set()
            if upload_thread is not None:
                upload_thread.join()

    # Updating the job status
    try:
        if is_completed:
            status_logging.get_status_logger().write(
                message=f"{job['action']} action completed successfully for {job['neural_network_name']}",
                status_level=status_logging.Status.SUCCESS
            )
        else:
            status_logging.get_status_logger().write(
                message=f"{job['action']} action failed for {job['neural_network_name']}",
                status_level=status_logging.Status.FAILURE
            )
    except Exception:
        print(traceback.format_exc())

    return is_completed