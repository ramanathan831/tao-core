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

"""Handler functions to manage NGC related operations"""
import io
import os
import requests
import zipfile

import logging
logger = logging.getLogger(__name__)

NUM_OF_RETRY = 3


def send_admin_get_request(endpoint, headers, retry=0):
    """Send an GET request with retries.

    Args:
        endpoint (str): The URL endpoint for the GET request.
        headers (dict): The headers to include in the GET request.
        retry (int, optional): The current retry attempt (default is 0).

    Returns:
        requests.Response: The response object from the GET request.

    Raises:
        requests.RequestException: If the GET request encounters an error after all retries.
    """
    r = requests.get(endpoint, headers=headers, timeout=3600)
    if not r.ok:
        if retry < NUM_OF_RETRY:
            logging.info("Retrying {} time(s) to GET {}.".format(retry, endpoint))  # noqa pylint: disable=C0209
            return send_admin_get_request(endpoint, headers, retry + 1)
        logging.info("Request to GET {} failed after {} retries.".format(endpoint, retry))  # noqa pylint: disable=C0209
    return r


def download_ngc_model(ngc_path, ptm_root, api_key, is_cookie_set=False):
    """Download models from NGC model registry.

    Args:
        ngc_path (str): The NGC path to the desired model in the format 'org/team/model:version'.
        ptm_root (str): The directory where the downloaded model will be saved.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    if ngc_path == "":
        logging.info("Invalid ngc path.")
        return False
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    model, version = ngc_configs[-1].split(':')
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]

    # Get access token using k8s admin secret
    if not api_key:
        logging.info("API key/Cookie is None")
        return False

    if is_cookie_set:
        headers = {'Accept': 'application/json', 'Cookie': api_key}
    else:
        headers = {'Accept': 'application/json', 'Authorization': 'ApiKey ' + api_key}
        response = send_admin_get_request("https://authn.nvidia.com/token?service=ngc", headers=headers)
        if not response.ok:
            logging.info("API response is not ok")
            return False
        token = response.json()["token"]
        headers = {"Authorization": f"Bearer {token} "}

    url_substring = ""
    if team and team != "no-team":
        url_substring = f"team/{team}"
    base_url = "https://api.ngc.nvidia.com"
    files_endpoint = f"v2/org/{org}/{url_substring}/models/{model}/versions/{version}/files".replace("//", "/")
    files_url = f"{base_url}/{files_endpoint}"
    logging.info("Calling NGC API to list base_experiment files {}".format(files_url))  # noqa pylint: disable=C0209
    response = send_admin_get_request(files_url, headers=headers)
    if not response.ok:
        logging.info("Download API response is not ok")
        return False

    file_info = response.json()
    dest_root = f"{ptm_root}/{model}_v{version}"

    for file_info in file_info['modelFiles']:
        path = file_info['path']
        download_endpoint = f"v2/org/{org}/{url_substring}/models/{model}/versions/{version}/files/{path}/zip/download".replace("//", "/")
        download_url = f"{base_url}/{download_endpoint}"
        download_response = send_admin_get_request(download_url, headers=headers)
        if download_response.status_code == 200:
            dest_path = os.path.join(dest_root, path)
            file_dir = os.path.dirname(dest_path)
            os.makedirs(file_dir, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(download_response.content)) as z:
                z.extractall(file_dir)
            logging.info("Saving base_experiment file to {}".format(dest_root))  # noqa pylint: disable=C0209
        else:
            logging.info("Failed to download {}. Status code: {}".format(path, response.status_code))  # noqa pylint: disable=C0209
            return False

    return True
