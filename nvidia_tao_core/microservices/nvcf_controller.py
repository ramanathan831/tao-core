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

"""NVCF job's kubernetes controller"""
import os
import sys
import json
import time
import asyncio
import traceback
from datetime import datetime
from kubernetes import client, config
from concurrent.futures import ThreadPoolExecutor

from nvidia_tao_core.microservices.handlers.nvcf_handler import (
    invoke_function,
    get_status_of_invoked_function,
    create_function,
    deploy_function,
    get_function,
    delete_function_version
)
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    update_job_message,
    update_job_details_with_microservices_response,
    update_status_json,
    get_log_file_path
)
from nvidia_tao_core.microservices.handlers.utilities import get_cloud_metadata
from nvidia_tao_core.microservices.utils import safe_load_file

if not os.getenv("CI_PROJECT_DIR", None):
    config.load_incluster_config()
api_instance = client.CustomObjectsApi()
executor = ThreadPoolExecutor(max_workers=10)

job_tracker = {}
logs_tracker = set([])
active_tasks = set([])


def internal_job_status_update(user_id, org_name, job_id, automl_experiment_number, message, logfile=""):
    """Post an status update to the job"""
    date_time = datetime.now()
    date_object = date_time.date()
    time_object = date_time.time()
    date = "{}:{}:{}".format(  # noqa pylint: disable=C0209
        time_object.hour,
        time_object.minute,
        time_object.second
    )
    time = "{}/{}/{}".format(  # noqa pylint: disable=C0209
        date_object.month,
        date_object.day,
        date_object.year
    )
    data = {
        "date": date,
        "time": time,
        "status": "FAILURE",
        "verbosity": "INFO",
    }
    if message:
        data["message"] = message
    data_string = json.dumps(data)
    callback_data = {
        "experiment_number": automl_experiment_number,
        "status": data_string,
    }
    update_status_json(user_id, org_name, job_id, callback_data)

    if logfile and os.path.exists(os.path.dirname(logfile)):
        with open(logfile, "a", encoding='utf-8') as f:
            f.write(f"\n{message}\n")


def update_cr_status(namespace, custom_resource_name, status):
    """Update status of the NVCF Custom resource"""
    updated_cr = api_instance.patch_namespaced_custom_object(
        group="nvcf-job-manager.nvidia.io",
        version="v1alpha1",
        namespace=namespace,
        plural="nvcfjobs",
        name=custom_resource_name,
        body={"status": {"phase": status}}
    )
    return updated_cr


def create_and_deploy_function_sync(org_name, team_name, job_id, container, nvcf_backend_details, ngc_key):
    """Create and deploy a NVCF function (blocking)"""
    try:
        create_response = create_function(org_name, team_name, job_id, container, ngc_key)
        if create_response.ok:
            print(f"Function created successfully for job {job_id}", file=sys.stderr)
            function_metadata = create_response.json()
            deploy_response = deploy_function(org_name, team_name, function_metadata, nvcf_backend_details, ngc_key)
            if deploy_response.ok:
                print(f"Function deployment initiated successfully for job {job_id}", file=sys.stderr)
                while True:
                    function_id = function_metadata["function"]["id"]
                    version_id = function_metadata["function"]["versionId"]
                    current_function_response = get_function(org_name, team_name, function_id, version_id, ngc_key)
                    if current_function_response.ok:
                        current_function_metadata = current_function_response.json()
                        if current_function_metadata.get("function", {}).get("status") == "ACTIVE":
                            deployment_string = (
                                f"{function_metadata['function']['id']}:"
                                f"{function_metadata['function']['versionId']}"
                            )
                            print(f"Function {deployment_string} for {job_id} deployed successfully", file=sys.stderr)
                            return deployment_string, f"Function {deployment_string} for {job_id} deployed successfully"
                        if current_function_metadata.get("function", {}).get("status") == "ERROR":
                            print(f"Get function deployment status for job {job_id} returned error", file=sys.stderr)
                            return "False", f"Get function deployment status for job {job_id} returned error"
                        print(
                            f"Function id {function_id} for job {job_id} status: "
                            f"{current_function_metadata.get('function', {}).get('status')}",
                            file=sys.stderr
                        )
                    else:
                        print(f"Get function deployment status for job {job_id} failed", file=sys.stderr)
                        return "False", f"Get function deployment status for job {job_id} failed"
                    time.sleep(10)
            else:
                print(f"Function deployment request failed for job {job_id}", file=sys.stderr)
                return "False", f"Function deployment request failed for job {job_id}"
        else:
            print(f"Function creation request failed for job {job_id}", file=sys.stderr)
            return "False", f"Function creation request failed for job {job_id}"
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"Error in create_and_deploy_function: {str(e)} for job {job_id}", file=sys.stderr)
        raise Exception(e) from e


async def create_and_deploy_function(org_name, team_name, job_id, container, nvcf_backend_details, ngc_key):
    """Create and deploy a NVCF function (non-blocking)"""
    loop = asyncio.get_event_loop()
    args = (org_name, team_name, job_id, container, nvcf_backend_details, ngc_key)
    deployment_string, message = await loop.run_in_executor(
        executor,
        create_and_deploy_function_sync,
        *args
    )
    return deployment_string, message


async def create_nvcf_job(nvcf_cr):
    """Construct requests call for triggering Job on NVCF cloud"""
    config.load_incluster_config()

    assert nvcf_cr.get("metadata", "")
    assert nvcf_cr.get("spec", "")

    namespace = nvcf_cr['metadata']['namespace']
    custom_resource_name = nvcf_cr['metadata']['name']

    user_id = nvcf_cr["spec"].get("user_id")
    org_name = nvcf_cr["spec"].get("org_name")
    team_name = nvcf_cr["spec"].get("team_name")
    job_id = nvcf_cr["spec"].get("job_id")
    job_handler_id = nvcf_cr["spec"].get("job_handler_id")
    job_kind = nvcf_cr["spec"].get("job_kind")
    action = nvcf_cr["spec"].get("action")
    network = nvcf_cr["spec"].get("network")
    deployment_string = nvcf_cr["spec"].get("deployment_string")
    container = nvcf_cr["spec"].get("container")
    tao_api_admin_key = nvcf_cr["spec"].get("tao_api_admin_key")
    tao_api_base_url = nvcf_cr["spec"].get("tao_api_base_url")
    tao_api_status_callback_url = nvcf_cr["spec"].get("tao_api_status_callback_url", "")
    tao_api_ui_cookie = nvcf_cr["spec"].get("tao_api_ui_cookie")
    use_ngc_staging = nvcf_cr["spec"].get("use_ngc_staging")
    automl_experiment_number = nvcf_cr["spec"].get("automl_experiment_number", "0")
    nvcf_backend_details = nvcf_cr["spec"].get("nvcf_backend_details")
    ngc_key = nvcf_cr["spec"].get("ngc_key")

    job_message_job_id = tao_api_status_callback_url.split("/")[-1]
    logfile = get_log_file_path(user_id, org_name, job_handler_id, job_message_job_id, job_id, automl_experiment_number)
    if not deployment_string:
        update_job_message(
            job_handler_id,
            job_message_job_id,
            job_kind,
            "NVCF function is being deployed",
            automl_expt_job_id=job_id,
            update_automl_expt=True
        )
        deployment_string, message = await create_and_deploy_function(
            org_name,
            team_name,
            job_id,
            container,
            nvcf_backend_details,
            ngc_key
        )
        if deployment_string == "False":
            internal_job_status_update(
                user_id,
                org_name,
                job_message_job_id,
                automl_experiment_number,
                f"{message}\nNVCF deployment errored out, retry job again",
                logfile
            )
            print(
                f"Setting customer resource {custom_resource_name} to Error as NVCF function can't be deployed",
                file=sys.stderr
            )
            print("Unable to deploy NVCF function, Retry TAO job", file=sys.stderr)
            return update_cr_status(namespace, custom_resource_name, "Error")

    updated_spec = {"deployment_string": deployment_string}
    # Patch the custom resource with the updated deployment_string
    updated_cr = api_instance.patch_namespaced_custom_object(
        group="nvcf-job-manager.nvidia.io",
        version="v1alpha1",
        namespace=namespace,
        plural="nvcfjobs",
        name=custom_resource_name,
        body={"spec": updated_spec},
    )

    cloud_metadata = {}
    get_cloud_metadata(nvcf_cr["spec"].get("workspace_ids"), cloud_metadata)

    spec_file_path = nvcf_cr["spec"].get("spec_file_path")
    if not spec_file_path:
        print("spec_file_path not set", file=sys.stderr)
        internal_job_status_update(
            user_id,
            org_name,
            job_message_job_id,
            automl_experiment_number,
            "Spec file couldn't be found during deployment of NVCF function",
            logfile
        )
        print(
            f"Setting customer resource {custom_resource_name} to Error as spec file can't be found",
            file=sys.stderr
        )
        return update_cr_status(namespace, custom_resource_name, "Error")
    specs = safe_load_file(spec_file_path, file_type="yaml")

    job_create_response = invoke_function(deployment_string,
                                          network,
                                          action,
                                          microservice_action="post_action",
                                          cloud_metadata=cloud_metadata,
                                          specs=specs,
                                          ngc_key=ngc_key,
                                          job_id=job_id,
                                          tao_api_admin_key=tao_api_admin_key,
                                          tao_api_base_url=tao_api_base_url,
                                          tao_api_status_callback_url=tao_api_status_callback_url,
                                          tao_api_ui_cookie=tao_api_ui_cookie,
                                          use_ngc_staging=use_ngc_staging,
                                          automl_experiment_number=automl_experiment_number)

    if job_create_response.status_code not in [200, 202]:
        job_create_response_json = job_create_response.json()
        print("Invocation error response code", job_create_response.status_code, file=sys.stderr)
        print("Invocation error response json", job_create_response_json, file=sys.stderr)
        update_job_details_with_microservices_response(
            job_create_response_json.get('detail', ""),
            job_message_job_id,
            automl_expt_job_id=job_id
        )
        print(
            f"Setting customer resource {custom_resource_name} to Error as microservices job couldn't be created",
            file=sys.stderr
        )
        return update_cr_status(namespace, custom_resource_name, "Error")

    job_create_response_json = job_create_response.json()
    print(
        f"Microservice job successfully created for {custom_resource_name}",
        job_create_response_json,
        file=sys.stderr
    )
    req_id = job_create_response_json.get("reqId", "")
    job_id = job_create_response_json.get("response", {}).get("job_id")

    if job_create_response.status_code == 202:
        while True:
            polling_response = get_status_of_invoked_function(req_id, ngc_key)
            if polling_response.status_code == 404:
                if polling_response.json().get("title") != "Not Found":
                    print("Polling(job_create) response failed", polling_response.status_code, file=sys.stderr)
                    internal_job_status_update(
                        user_id,
                        org_name,
                        job_message_job_id,
                        automl_experiment_number,
                        "NVCF Polling failed",
                        logfile
                    )
                    print(
                        f"Setting customer resource {custom_resource_name} to Error as job create polling failed",
                        file=sys.stderr
                    )
                    return update_cr_status(namespace, custom_resource_name, "Error")
            if polling_response.status_code != 202:
                break
            time.sleep(10)

        if polling_response.status_code != 200:
            print("Polling(job_create) response status code is not 200", polling_response.status_code, file=sys.stderr)
            internal_job_status_update(
                user_id,
                org_name,
                job_message_job_id,
                automl_experiment_number,
                "NVCF Polling failed",
                logfile
            )
            error_msg = (
                f"Setting customer resource {custom_resource_name} "
                "to Error as job create polling failed with a non 200 response"
            )
            print(error_msg, file=sys.stderr)
            return update_cr_status(namespace, custom_resource_name, "Error")
        job_id = polling_response.json().get("response", {}).get("job_id")

    if not job_id:
        print("Job ID couldn't be fetched", file=sys.stderr)
        internal_job_status_update(
            user_id,
            org_name,
            job_message_job_id,
            automl_experiment_number,
            "Job_id from microservices job created couldn't be fetched",
            logfile
        )
        print(
            f"Setting customer resource {custom_resource_name} to Error as job id can't be fetched from microservices",
            file=sys.stderr
        )
        return update_cr_status(namespace, custom_resource_name, "Error")

    return updated_cr


def delete_nvcf_job(nvcf_cr):
    """Construct requests call for deleting Job on NVCF cloud"""
    deployment_string = nvcf_cr["spec"].get("deployment_string")
    org_name = nvcf_cr["spec"].get("org_name")
    team_name = nvcf_cr["spec"].get("team_name")
    ngc_key = nvcf_cr["spec"].get("ngc_key")
    if deployment_string.find(":") == -1:
        print(f"Deployment not active yet for custom resource {nvcf_cr['metadata']['name']}", file=sys.stderr)
        return
    function_id, version_id = deployment_string.split(":")
    if org_name not in ["0544357712065245"]:
        delete_function_version(org_name, team_name, function_id, version_id, ngc_key)


def get_job_logs(user_id, job_id, orgName):
    """Get job logs from BCP"""
    return


def print_job_logs(user_id, job_id, orgName, custom_resource_name):
    """Print logs of NVCF job on controller pod"""
    return


def overwrite_job_logs_from_bcp(logfile, job_name):
    """Get job logs from BCP and overwrite it with existing logs"""
    return


def get_nvcf_job_status(nvcf_cr, status="", function_id="", version_id=""):
    """Get and update NVCF custom resource status"""
    custom_resource_name = nvcf_cr.get("metadata", {}).get('name')
    org_name = nvcf_cr["spec"].get("org_name")
    team_name = nvcf_cr["spec"].get("team_name")
    ngc_key = nvcf_cr["spec"].get("ngc_key")
    if not status:
        namespace = nvcf_cr['metadata']['namespace']

        user_id = nvcf_cr["spec"].get("user_id")
        action = nvcf_cr["spec"].get("action")
        network = nvcf_cr["spec"].get("network")
        job_id = nvcf_cr["spec"].get("job_id")
        job_handler_id = nvcf_cr["spec"].get("job_handler_id")
        automl_experiment_number = nvcf_cr["spec"].get("automl_experiment_number", "0")
        tao_api_status_callback_url = nvcf_cr["spec"].get("tao_api_status_callback_url", "")

        job_message_job_id = tao_api_status_callback_url.split("/")[-1]

        deployment_string = nvcf_cr["spec"].get("deployment_string")
        if deployment_string.find(":") == -1:
            if nvcf_cr.get("status", {}).get("phase", "") == "Error":
                print(
                    f"Returning NVCF job status as error for {custom_resource_name} "
                    "because phase is error and deployment string is not valid",
                    file=sys.stderr
                )
                return "Error"
            print(
                f"Deployment not active yet for job {job_id} {deployment_string} (in get status function)",
                file=sys.stderr
            )
            status = "Pending"
            return status

        if nvcf_cr.get("status", {}).get("phase", "") in ("Done", "Error"):
            if nvcf_cr.get("status", {}).get("phase", "") == "Error":
                print(
                    f"Returning NVCF job status as error for {custom_resource_name} because phase is error",
                    file=sys.stderr
                )
            return nvcf_cr.get("status", {}).get("phase", "")

        function_id, version_id = deployment_string.split(":")

        print("update status", deployment_string, job_tracker.keys(), file=sys.stderr)
        job_monitor_response = invoke_function(
            deployment_string,
            network,
            action,
            microservice_action="get_job_status",
            ngc_key=ngc_key,
            job_id=job_id
        )
        if job_monitor_response.status_code == 404:
            status = "Error"
            if job_monitor_response.json().get("title") == "Not Found":
                print(f"NVCF function was deleted, setting status as done for {custom_resource_name}", file=sys.stderr)
                status = "Done"

        if job_monitor_response.status_code == 202:
            req_id = job_monitor_response.json().get("reqId", "")
            while True:
                job_monitor_response = get_status_of_invoked_function(req_id, ngc_key)
                if job_monitor_response.status_code == 404:
                    if job_monitor_response.json().get("title") != "Not Found":
                        print("Polling(job_monitor) response failed", job_monitor_response.status_code, file=sys.stderr)
                        status = "Error"
                        print(
                            f"Setting NVCF job status as error for {custom_resource_name} "
                            "because status polling failed",
                            file=sys.stderr
                        )
                if job_monitor_response.status_code != 202:
                    break
                time.sleep(10)

            if job_monitor_response.status_code != 200:
                print(
                    "Polling(job_monitor) response status code is not 200",
                    job_monitor_response.status_code,
                    file=sys.stderr
                )
                print(
                    f"Setting NVCF job status as error for {custom_resource_name} "
                    "because status polling failed with non 200 response",
                    file=sys.stderr
                )
                status = "Error"

        if not status:
            try:
                job_monitor_response_json = job_monitor_response.json()
                error_message = job_monitor_response_json.get("response", {}).get("detail")
                status = job_monitor_response_json.get("response", {}).get("status")
                if status:
                    if status == "Processing":
                        status = "Running"
                    elif status not in ("Pending", "Done"):
                        logfile = get_log_file_path(
                            user_id,
                            org_name,
                            job_handler_id,
                            job_message_job_id,
                            job_id,
                            automl_experiment_number
                        )
                        internal_job_status_update(
                            user_id,
                            org_name,
                            job_message_job_id,
                            automl_experiment_number,
                            "Container microservices reported an error, more logs to be found on NVCF UI",
                            logfile
                        )
                        print(
                            f"Setting NVCF job status as error for {custom_resource_name} "
                            f"because status from microservices is {status}",
                            file=sys.stderr
                        )
                        status = "Error"
                else:
                    status = "Pending"
                    if "Job ID Not Present" in error_message:
                        print(
                            f"Job ID Not Present in {deployment_string} for job {job_id}",
                            file=sys.stderr
                        )
                        print(
                            f"Setting NVCF job status as error for {custom_resource_name} "
                            "because Job is not present in microservices",
                            file=sys.stderr
                        )
                        status = "Error"
            except Exception as e:
                print(f"Exception thrown in get_nvcf_job_status is {str(e)}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                print(
                    f"Exception while calling job fetch microservices in {deployment_string} for job {job_id}, "
                    f"{job_monitor_response.text}",
                    file=sys.stderr
                )
                print(
                    f"Setting NVCF job status as error for {custom_resource_name} because of run time exception",
                    file=sys.stderr
                )
                status = "Error"

    if status in ("Done", "Error"):
        print(
            f"Status is {status}. Hence, deleting the function {function_id} with version {version_id} "
            f"for {custom_resource_name}",
            file=sys.stderr
        )
        if function_id and version_id:
            if org_name not in ["0544357712065245"]:
                delete_function_version(org_name, team_name, function_id, version_id, ngc_key)

    if not status:
        print(f"Status couldn't be inferred for {custom_resource_name}", file=sys.stderr)
        status = "Pending"
    try:
        namespace = nvcf_cr['metadata']['namespace']
        update_cr_status(namespace, custom_resource_name, status)
    except Exception:
        pass
    return status


def update_status(job_tracker, logs_tracker):
    """Update the status of the custom resources based on status of NGC job"""
    for _, nvcf_cr in job_tracker.items():
        custom_resource_name = nvcf_cr["metadata"].get('name')
        status = get_nvcf_job_status(nvcf_cr)

        if status in ("Done", "Error"):
            logs_tracker.add(custom_resource_name)
            print(f"Status is {status} for {custom_resource_name}. Hence, deleting the function", file=sys.stderr)
            continue
        job_id = nvcf_cr["spec"].get("job_id")
        deployment_string = nvcf_cr["spec"].get("deployment_string")
        if deployment_string.find(":") == -1:
            print(
                f"Deployment not active yet for job {job_id} {deployment_string} (in update status function)",
                file=sys.stderr
            )
            continue


def remove_deleted_custom_resources(job_tracker, logs_tracker):
    """Remove deleted custom resources from tracker variables"""
    try:
        # Fetch the list of NVCF job custom resources
        nvcf_jobs = api_instance.list_cluster_custom_object(
            group="nvcf-job-manager.nvidia.io",
            version="v1alpha1",
            plural="nvcfjobs",
        )

        existing_jobs = set(job_tracker.keys())
        current_jobs = set(item['metadata']['name'] for item in nvcf_jobs['items'])
        deleted_jobs = existing_jobs - current_jobs

        for deleted_job in deleted_jobs:
            print(f"NVCF CR deleted: {deleted_job}")
            delete_nvcf_job(job_tracker[deleted_job])
            del job_tracker[deleted_job]

    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"Error in removing custom resource from tracker: {str(e)}", file=sys.stderr)


async def handle_new_nvcf_job(custom_resource_name, item):
    """Handle new NVCF job creation and deployment."""
    global job_tracker  # pylint: disable=global-statement
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global active_tasks  # pylint: disable=global-statement
    active_tasks = active_tasks if 'active_tasks' in globals() else set()
    # Check if the job is already being processed
    if custom_resource_name not in job_tracker:
        print(f"Job {custom_resource_name} is already being processed", file=sys.stderr)
        return
    updated_item = await create_nvcf_job(item)
    if updated_item is not None:
        job_tracker[custom_resource_name] = updated_item
    active_tasks.remove(asyncio.current_task())


async def process_events():
    """Process NVCF JOB events"""
    global job_tracker  # pylint: disable=global-statement
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global logs_tracker  # pylint: disable=global-statement
    logs_tracker = logs_tracker if 'logs_tracker' in globals() else {}
    global active_tasks  # pylint: disable=global-statement
    active_tasks = active_tasks if 'active_tasks' in globals() else set()

    while True:
        try:
            # Fetch the list of NVCF job custom resources
            nvcf_jobs = api_instance.list_cluster_custom_object(
                group="nvcf-job-manager.nvidia.io",
                version="v1alpha1",
                plural="nvcfjobs",
            )

            for item in nvcf_jobs['items']:
                custom_resource_name = item['metadata']['name']

                if custom_resource_name not in job_tracker:
                    # Handle added event
                    print(f"NVCF CR added: {custom_resource_name}", file=sys.stderr)
                    task = asyncio.create_task(handle_new_nvcf_job(custom_resource_name, item))
                    job_tracker[custom_resource_name] = item
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

            remove_deleted_custom_resources(job_tracker, logs_tracker)
            await asyncio.sleep(10)

        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            print(f"Error in the event processing loop: {str(e)}", file=sys.stderr)


async def main():
    """Controller Main function"""
    global job_tracker  # pylint: disable=global-statement
    job_tracker = job_tracker if 'job_tracker' in globals() else {}
    global logs_tracker  # pylint: disable=global-statement
    logs_tracker = logs_tracker if 'logs_tracker' in globals() else {}
    global active_tasks  # pylint: disable=global-statement
    active_tasks = active_tasks if 'active_tasks' in globals() else set()

    asyncio.create_task(process_events())
    while True:
        remove_deleted_custom_resources(job_tracker, logs_tracker)
        update_status(job_tracker, logs_tracker)
        await asyncio.sleep(10)

if __name__ == "__main__":
    # Run the main function asynchronously
    asyncio.run(main())
