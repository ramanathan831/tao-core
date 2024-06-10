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

"""Utilties to send data to the TAO Toolkit Telemetry Remote Service."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib

import requests
import urllib3

# TODO @vpraveen: This must be removed at the time of release we figure out where to
# host this.
TAO_CERTIFICATES_URL = "https://gitlab-master.nvidia.com/vpraveen/python_wheels/-/raw/main/certs/certificates.tar.gz"  # noqa pylint: disable=E501
TAO_SERVER_IP = "10.111.60.42"


def get_server_url():
    """Get the Telemetry Server URL."""
    url = os.getenv("TAO_TELEMETRY_SERVER", None)
    if url is None:
        url = 'https://tao-telemetry.nvidia.com:9443/api/v1/telemetry'
    return url


def url_exists(url):
    """Check if a URL exists.

    Args:
        url (str): String to be verified as a URL.

    Returns:
        valid (bool): True/Falso
    """
    url_request = urllib.request.Request(url)
    url_request.get_method = lambda: 'HEAD'
    try:
        with urllib.request.urlopen(url_request):
            return True
    except urllib.request.HTTPError:
        return False


def get_certificates():
    """Download the cacert.pem file and return the path.

    Returns:
        path (str): UNIX path to the certificates.
    """
    certificates_url = os.getenv("TAO_CERTIFICATES_URL", TAO_CERTIFICATES_URL)
    if not url_exists(certificates_url):
        raise urllib.request.HTTPError("Url for the certificates not found.")
    tmp_dir = tempfile.mkdtemp()
    download_command = f"wget {certificates_url} -P {tmp_dir} --quiet"
    try:
        subprocess.check_call(
            download_command, shell=True, stdout=sys.stdout
        )
    except subprocess.CalledProcessError as exc:
        raise urllib.request.HTTPError("Download certificates.tar.gz failed.") from exc
    tarfile_path = os.path.join(tmp_dir, "certificates.tar.gz")
    assert tarfile.is_tarfile(tarfile_path), (
        "The downloaded file isn't a tar file."
    )
    with tarfile.open(name=tarfile_path, mode="r:gz") as tar_file:
        filenames = tar_file.getnames()
        for memfile in filenames:
            member = tar_file.getmember(memfile)
            tar_file.extract(member, tmp_dir)
    file_list = [item for item in os.listdir(tmp_dir) if item.endswith(".pem")]
    assert file_list, (
        f"Didn't get pem files. Directory contents {file_list}"
    )
    return tmp_dir


def send_telemetry_data(network, action, gpu_data, num_gpus=1, time_lapsed=None):
    """Wrapper to send TAO telemetry data.

    Args:
        network (str): Name of the network being run.
        action (str): Subtask of the network called.
        gpu_data (dict): Dictionary containing data about the GPU's in the machine.
        num_gpus (int): Number of GPUs used in the job.
        time_lapsed (int): Time lapsed.

    Returns:
        No explicit returns.
    """
    urllib3.disable_warnings(urllib3.exceptions.SubjectAltNameWarning)
    if os.getenv('TELEMETRY_OPT_OUT', "no").lower() in ["no", "false", "0"]:
        url = get_server_url()
        data = {
            "version": os.getenv("TAO_TOOLKIT_VERSION", "4.22.11"),
            "action": action,
            "network": network,
            "gpu": [device["name"] for device in gpu_data[:num_gpus]],
            "gpu_details": gpu_data[:num_gpus]
        }
        if time_lapsed is not None:
            data["time_lapsed"] = time_lapsed
        certificate_dir = get_certificates()
        cert = ('client-cert.pem', 'client-key.pem')
        verify = 'ca-cert.pem'
        requests.post(
            url,
            json=data,
            cert=tuple((os.path.join(certificate_dir, item) for item in cert)),
            verify=os.path.join(certificate_dir, verify),
            timeout=10
        )
        print(f"Telemetry data posted: \n{json.dumps(data, indent=4)}")
        shutil.rmtree(certificate_dir)


if __name__ == "__main__":
    print("Send dummy data.")
    send_telemetry_data(
        "detectnet_v2",
        "train",
        "gpu_data",
        1
    )
