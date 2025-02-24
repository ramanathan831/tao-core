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
from ngcbpc import errors

import os
import requests

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


def split_ngc_path(ngc_path):
    """Split ngc path into org, team and model name, model version"""
    path_split = ngc_path.replace("/no-team", "").split("/")
    if len(path_split) == 3:
        org, team, model_name = path_split
    elif len(path_split) == 2:
        org, model_name = path_split
        team = ""
    else:
        raise ValueError(f"Invalid ngc_path: {ngc_path}")
    if ":" in model_name:
        model_name, model_version = model_name.split(":")
    else:
        model_version = ""
    return org, team, model_name, model_version


def download_ngc_model(ngc_path, ptm_root, key, is_cookie_set, use_ngc_staging):
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
    if not key.startswith("nvapi"):
        logging.info('Credentials error: Invalid NGC_PERSONAL_KEY, NGC_keys are no longer valid, generate a personal key with Cloud Functions, NGC Catalog and Private registry services https://org.ngc.nvidia.com/setup/personal-keys')
        return False
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]

    # Get access token using k8s admin secret
    if not key:
        logging.info("Personal key/Cookie is None")
        return False

    # Download model with ngc sdk
    from ngcsdk import Client  # pylint: disable=C0415
    clt = Client()

    try:
        clt.configure(api_key=key, org_name=org, team_name=team)
    except Exception as e:
        if not ("Invalid org" in str(e) or "Invalid team" in str(e)):
            logging.error("Can't configure the passed NGC KEY for Org {}, team {}".format(org, team)) # noqa pylint: disable=C0209
            return False
        logging.info("Can't validate the passed NGC KEY for Org {}, team {}, going to try download without configuring credentials".format(org, team)) # noqa pylint: disable=C0209
    try:
        if not os.path.exists(ptm_root):
            os.makedirs(ptm_root, exist_ok=True)
            clt.registry.model.download_version(ngc_path, destination=ptm_root)
            logging.info("Saving base_experiment file to {}".format(ptm_root)) # noqa pylint: disable=C0209
        else:
            logging.info("Base_experiment already present in {}".format(ptm_root)) # noqa pylint: disable=C0209
    except errors.ResourceNotFoundException as e:
        logging.error("Model {} not found. Error: {}".format(ngc_path, e))  # noqa pylint: disable=C0209
        return False
    except errors.NgcException as e:
        logging.error("Failed to download {}. Error: {}".format(ngc_path, e))  # noqa pylint: disable=C0209
        return False

    return True
