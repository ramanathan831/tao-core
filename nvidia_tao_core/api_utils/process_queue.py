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


import os
import re
import sys
import yaml
import shlex
import importlib
import subprocess
import threading
import traceback
from time import time, sleep
from contextlib import contextmanager

from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.cloud_handlers.utils import download_files_from_spec, get_results_cloud_data, monitor_and_upload
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points

module = entry_points[0].module_name.split('.')[0] if entry_points else None
entrypoint = importlib.import_module(entrypoint_paths[module]) if module else None

# Initialize empty queue, processing jobs, and completed jobs lists
queue = []
processing_jobs = []
completed_jobs = []


def convert_dict_to_cli_args(data, parent_key=""):
    cli_args = []
    for key, value in data.items():
        # Construct the current key path
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            cli_args.extend(convert_dict_to_cli_args(value, key))
        else:
            # Append the CLI argument as --key value
            cli_args.append(f"--{key}")
            cli_args.append(str(value))
    
    return cli_args


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, "a") as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def vlm_launch(neural_network_name, action, specs):
    cli_args = convert_dict_to_cli_args(specs)
    cli_args = " ".join(cli_args)
    process_passed = False
    try:
        # Run the script.
        log_file = ""
        if os.getenv("JOB_ID"):
            logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
            log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

        progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")
        call = f"{neural_network_name}-{action} {cli_args}"
        start = time()
        print("call", call)
        with dual_output(log_file) as (stdout_target, log_target):
            proc = subprocess.Popen(
                shlex.split(call),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            )
            last_progress_bar_line = None

            for line in proc.stdout:
                # Check if the line contains \r or matches the progress bar pattern
                if '\r' in line or progress_bar_pattern.search(line):
                    last_progress_bar_line = line.strip()
                    # Print the progress bar line to the terminal
                    stdout_target.write('\r' + last_progress_bar_line)
                    stdout_target.flush()
                else:
                    # Write the final progress bar line to the log file before a new log line
                    if last_progress_bar_line:
                        if log_target:
                            log_target.write(last_progress_bar_line + '\n')
                            log_target.flush()
                        last_progress_bar_line = None
                    stdout_target.write(line)
                    stdout_target.flush()
                    if log_target:
                        log_target.write(line)
                        log_target.flush()

            proc.wait()  # Wait for the process to complete
            # Write the final progress bar line after process completion
            if last_progress_bar_line and log_target:
                log_target.write(last_progress_bar_line + '\n')
                log_target.flush()
            if proc.returncode == 0:
                process_passed = True

    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted")
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        process_passed = False

    end = time()
    time_lapsed = int(end - start)
    if not process_passed:
        print("Execution status: FAIL")
        return False

    print("Execution status: PASS")
    return True


# Function to process jobs from the queue
def process_queue():
    """Scans for new jobs and pushes them to running state"""
    while True:
        if queue:
            job = queue.pop(0)  # Take the first job from the queue
            processing_jobs.append(job)  # Add job to processing list
            job['status'] = 'Processing'
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
                print("cloud storage", cloud_storage)
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

                print("entrypoint", entrypoint)
                if entrypoint:
                    _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                    is_completed = entrypoint.launch(args, "", actions, network=job["neural_network_name"])
                else:
                    is_completed = vlm_launch(job["neural_network_name"], job["action"], specs)

            except Exception:
                print("Traceback", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                job['status'] = 'Error'
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
                    job['status'] = 'Done'
                    status_logging.get_status_logger().write(
                        message=f"{job['action']} action completed successfully for {job['neural_network_name']}",
                        status_level=status_logging.Status.SUCCESS
                    )
                else:
                    job['status'] = 'Error'
                    status_logging.get_status_logger().write(
                        message=f"{job['action']} action failed for {job['neural_network_name']}",
                        status_level=status_logging.Status.FAILURE
                    )
            except Exception:
                print(traceback.format_exc())
                job['status'] = 'Error'

            processing_jobs.remove(job)  # Remove job from processing list
            completed_jobs.append(job)  # Add job to completed list
        sleep(1)  # Check queue every second
