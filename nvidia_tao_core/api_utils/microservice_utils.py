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

"""Microservice utils"""

import requests
import json


def invoke_microservices(request_dict):
    """Forward the superendpoint microservice request to the actual endpoint"""
    url = "http://localhost:8000/api/v1"
    api_endpoint = request_dict.get('api_endpoint', None)
    neural_network_name = request_dict.get('neural_network_name', None)
    ngc_key = request_dict.get('ngc_key', None)
    action_name = request_dict.get('action_name', None)
    storage = request_dict.get('storage', None)
    specs = request_dict.get('specs', None)
    job_id = request_dict.get('job_id', None)

    telemetry_opt_out = request_dict.get('telemetry_opt_out', "no")
    use_ngc_staging = request_dict.get('use_ngc_staging', "False")
    tao_api_ui_cookie = request_dict.get('tao_api_ui_cookie', "")
    tao_api_admin_key = request_dict.get('tao_api_admin_key', "")
    tao_api_base_url = request_dict.get('tao_api_base_url', "https://nvidia.com")
    tao_api_status_callback_url = request_dict.get('tao_api_status_callback_url', "https://nvidia.com")
    automl_experiment_number = request_dict.get('automl_experiment_number', "")
    hosted_service_interaction = request_dict.get('hosted_service_interaction', "")

    response = None
    if api_endpoint == "get_networks":
        response = requests.get(f"{url}/neural_networks")   # noqa pylint: disable=W3101
    elif api_endpoint == "get_actions":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions")   # noqa pylint: disable=W3101
    elif api_endpoint == "list_ptms":
        req_obj = {"ngc_key": ngc_key}
        response = requests.post(f"{url}/neural_networks/{neural_network_name}/pretrained_models", req_obj)   # noqa pylint: disable=W3101
    elif api_endpoint == "get_schema":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}:schema")   # noqa pylint: disable=W3101
    elif api_endpoint == "post_action":
        req_obj = {"specs": specs,
                   "cloud_metadata": storage,
                   "ngc_key": ngc_key,
                   "job_id": job_id,
                   "telemetry_opt_out": telemetry_opt_out,
                   "use_ngc_staging": use_ngc_staging,
                   "tao_api_ui_cookie": tao_api_ui_cookie,
                   "tao_api_admin_key": tao_api_admin_key,
                   "tao_api_base_url": tao_api_base_url,
                   "tao_api_status_callback_url": tao_api_status_callback_url,
                   "automl_experiment_number": automl_experiment_number,
                   "hosted_service_interaction": hosted_service_interaction,
                   }
        response = requests.post(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}", data=json.dumps(req_obj))   # noqa pylint: disable=W3101
    elif api_endpoint == "get_jobs":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}:ids")   # noqa pylint: disable=W3101
    elif api_endpoint == "get_job_status":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}/{job_id}")   # noqa pylint: disable=W3101

    if response and response.status_code in (200, 201):
        return response.json()

    raise ValueError(f"{response.json()['error_desc']}" if response.json().get('error_desc') else f"Failed to get execute (Status Code: {response.status_code} : {response.json()})")
