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
# See the License for the specific governing permissions and
# limitations under the License.
"""Handler to execute jobs on Lepton"""

import json
import os
import time
import uuid
from leptonai.api.v1.types.deployment import (
    LeptonDeployment,
    LeptonDeploymentUserSpec,
    LeptonContainer,
    ContainerPort,
    ResourceRequirement,
    LeptonLog,
    EnvVar
)
from leptonai.api.v1.types.job import (
    LeptonJob,
    LeptonJobUserSpec
)
from leptonai.api.v1.types.common import Metadata, LeptonVisibility
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
import logging
import requests
from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
from nvidia_tao_core.microservices.utils.handler_utils import send_microservice_request
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND, get_handler_metadata
from nvidia_tao_core.microservices.utils import get_admin_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LEPTON_API_BASE_URL = "https://gateway.dgxc-lepton.nvidia.com"


def get_lepton_handler_from_workspace(workspace_id):
    """Get the Lepton handler from the workspace"""
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
    lepton_workspace_id = cloud_specific_details.get("lepton_workspace_id")
    lepton_auth_token = cloud_specific_details.get("lepton_auth_token")
    if lepton_workspace_id and lepton_auth_token:
        try:
            logger.info(f"Instantiating Lepton Handler for workspace {workspace_id}")
            return LeptonHandler(lepton_workspace_id, lepton_auth_token)
        except Exception as e:
            logger.error(f"Exception instantiating Lepton Handler: {e}")
    return None


class LeptonHandler:
    """Handler to execute jobs on Lepton"""

    def __init__(self, workspace_id, auth_token):
        """Initialize the Lepton Handler"""
        self.workspace_id = workspace_id
        self.auth_token = auth_token
        self.api_client = APIClient(workspace_id=workspace_id, auth_token=auth_token)

    def get_lepton_job_name(self, job_id):
        """Get the Lepton job name"""
        # Lepton job name is limited to 36 characters and must start with alphabetical character
        return f"tao-{job_id}"[:36]

    def get_shapes_for_node_group(self, node_group):
        """Get the possible shapes for a dedicated node group"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/shapes"
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            params = {"node_groups": node_group}
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Failed to get shapes for node group {node_group}: {response.json()}")
        except Exception as e:
            logger.error(f"Error in get_shapes_for_node_group for {node_group}: {str(e)}")
        return []

    def list_image_pull_secrets(self):
        """List the image pull secrets"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/imagepullsecrets"
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = requests.get(endpoint, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Failed to list image pull secrets: {response.json()}")
        except Exception as e:
            logger.error(f"Error in list_image_pull_secrets: {str(e)}")
        return []

    def get_tao_image_pull_secret(self):
        """Get the TAO image pull secret"""
        secrets = self.list_image_pull_secrets()
        for secret in secrets:
            if secret.get("metadata", {}).get("id") == "tao-image-pull-secret":
                return "tao-image-pull-secret"
        logger.info("No TAO image pull secret found, creating one")
        ptm_key = get_admin_key(legacy_key=True)
        return self.create_image_pull_secret(ptm_key)

    def create_image_pull_secret(self, api_key, username="$oauthtoken",
                                 secret_name="tao-image-pull-secret",
                                 registry_server="https://nvcr.io"):
        """Create a image pull secret"""
        try:
            endpoint = f"{LEPTON_API_BASE_URL}/api/v2/workspaces/{self.workspace_id}/imagepullsecrets"
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "origin": LEPTON_API_BASE_URL,
            }
            data = {
                "metadata": {
                    "name": secret_name,
                    "id": secret_name
                },
                "spec": {
                    "registry_server": registry_server,
                    "username": username,
                    "password": api_key
                }
            }
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            if response.status_code == 201:
                logger.info("Image pull secret created successfully")
                return "tao-image-pull-secret"
            logger.error(f"Response: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error in create_image_pull_secret: {str(e)}")
        return None

    def get_available_lepton_instances(self):
        """Get the available Lepton instances"""
        available_lepton_instances = {}
        node_groups = self.api_client.nodegroup.list_all()
        for node_group in node_groups:
            resource_shapes = self.get_shapes_for_node_group(node_group.metadata.id_)
            for resource_shape in resource_shapes:
                spec = resource_shape.get('spec', {})
                metadata = resource_shape.get('metadata', {})
                accelerator_type = spec.get('accelerator_type')
                listable_in = spec.get('listable_in', [])
                if not listable_in or 'deployment' not in listable_in or not accelerator_type:
                    continue
                instance_id = metadata.get('id')
                platform_id = str(uuid.uuid5(
                    uuid.NAMESPACE_X500,
                    f"{node_group.metadata.id_}_{instance_id}_{accelerator_type}"))
                available_lepton_instances[platform_id] = {
                    "cluster": node_group.metadata.id_,
                    "gpu_type": instance_id,
                    "instance_type": accelerator_type,
                    "gpu_count": spec.get('accelerator_num'),
                    "cpu_cores": spec.get('cpu'),
                    "system_memory": f"{spec.get('memory_in_mb', 0) / 1024}GB",
                    "gpu_memory": f"{spec.get('accelerator_memory_in_mb', 0) / 1024}GB",
                    "backend_type": "lepton"
                }
        return available_lepton_instances

    def create_deployment(self, job_id, image, command, container_port=8000,
                          resource_shape=None, dedicated_node_group=None):
        """Create a Lepton deployment"""
        secret = self.get_tao_image_pull_secret()
        if secret:
            image_pull_secrets = [secret]
        else:
            logger.error("Failed to get TAO image pull secret, attempting deployment without it")
            image_pull_secrets = []
        if dedicated_node_group:
            logger.info(f"Creating deployment with dedicated node group {dedicated_node_group} "
                        f"and resource shape {resource_shape}")
            affinity = LeptonResourceAffinity(allowed_dedicated_node_groups=[dedicated_node_group])
        else:
            affinity = None
        lepton_deployment = LeptonDeployment(
            metadata=Metadata(name=self.get_lepton_job_name(job_id)),
            spec=LeptonDeploymentUserSpec(
                container=LeptonContainer(
                    image=image,
                    command=command,
                    ports=[ContainerPort(container_port=container_port)]),
                resource_requirement=ResourceRequirement(
                    resource_shape=resource_shape,
                    min_replicas=1,
                    max_replicas=1,
                    affinity=affinity
                ),
                image_pull_secrets=image_pull_secrets
            )
        )
        self.api_client.deployment.create(lepton_deployment)
        logger.info(f"Lepton Deployment {job_id} created successfully")

    def get_deployment(self, job_id):
        """Get deployment details by job_id."""
        deployment = self.api_client.deployment.get(self.get_lepton_job_name(job_id))
        return deployment

    def delete_deployment(self, job_id):
        """Delete deployment by job_id."""
        self.api_client.deployment.delete(self.get_lepton_job_name(job_id))
        logger.info(f"Lepton Deployment {job_id} deleted successfully")

    def scale_down_deployment(self, job_id, min_replicas=0):
        """Scale down deployment to minimum replicas."""
        spec = LeptonDeployment(
            spec=LeptonDeploymentUserSpec(
                resource_requirement=ResourceRequirement(
                    min_replicas=min_replicas,
                    max_replicas=min_replicas
                )
            )
        )
        self.api_client.deployment.update(self.get_lepton_job_name(job_id), spec)
        logger.info(f"Lepton Deployment {job_id} scaled down successfully")

    def scale_up_deployment(self, job_id, max_replicas=1):
        """Scale up deployment to specified max replicas."""
        spec = LeptonDeployment(
            spec=LeptonDeploymentUserSpec(
                resource_requirement=ResourceRequirement(
                    min_replicas=1,
                    max_replicas=max_replicas
                )
            )
        )
        self.api_client.deployment.update(self.get_lepton_job_name(job_id), spec)
        logger.info(f"Lepton Deployment {job_id} scaled up successfully")

    def wait_for_deployment(self, job_id):
        """Wait for deployment to become ready."""
        timeout = 300  # 5 minutes
        while True:
            deployment = self.get_deployment(job_id)
            deployment_status = deployment.status.state
            if deployment_status == "Ready":
                break
            time.sleep(5)
            timeout -= 5
            if timeout <= 0:
                logger.error(f"Deployment {job_id} timed out waiting for status: {deployment_status}")
                return None
        return deployment

    def get_node_group(self, node_group_id):
        """Get a node group"""
        node_groups = self.api_client.nodegroup.list_all()
        for node_group in node_groups:
            if node_group.metadata.id_ == node_group_id:
                return node_group
        return None

    def create_job(
            self,
            image,
            network,
            action,
            cloud_metadata={},
            specs={},
            job_id="",
            docker_env_vars={},
            resource_shape=None,
            dedicated_node_group=None,
            num_nodes=1,
    ):
        """Create a job on Lepton"""
        command = []
        if network == "cosmos-rl" and BACKEND == "local-docker":
            node_group = self.get_node_group(dedicated_node_group)
            if node_group:
                specs["lepton_specs"] = {
                    "lepton-job-name": self.get_lepton_job_name(job_id),
                    "lepton-container-image": image,
                    "lepton-resource-shape": resource_shape,
                    "lepton-node-group": node_group.metadata.name,
                    "lepton-image-pull-secrets": self.get_tao_image_pull_secret()
                }
            else:
                logger.error(f"Node group {dedicated_node_group} not found")
                raise Exception(f"Node group {dedicated_node_group} not found")
        if num_nodes > 1 and network != "cosmos-rl":
            command += [
                "wget -O init.sh https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh",
                "chmod +x init.sh",
                "source init.sh",
            ]
        command.append(
            "python -m nvidia_tao_core.microservices.handlers.container_handler "
            f"--neural-network-name {network} "
            f"--action-name {action} "
            f"--job-id {job_id} "
            f"--specs '{json.dumps(specs)}' "
            f"--docker-env-vars '{json.dumps(docker_env_vars)}'"
        )
        if network != "cosmos-rl":
            command = "; ".join(command)
        else:
            docker_env_vars["LEPTON_WORKSPACE_ID"] = self.workspace_id
            docker_env_vars["LEPTON_WORKSPACE_TOKEN"] = self.auth_token
            docker_handler = DockerHandler(image)
            docker_handler.start_container(
                container_name=self.get_lepton_job_name(job_id),
                command=command,
                docker_env_vars=docker_env_vars,
                num_gpus=0,
            )
            return None

        secret = self.get_tao_image_pull_secret()
        if secret:
            image_pull_secrets = [secret]
        else:
            logger.error("Failed to get TAO image pull secret, attempting deployment without it")
            image_pull_secrets = []
        if dedicated_node_group:
            logger.info(f"Creating job with dedicated node group {dedicated_node_group} "
                        f"and resource shape {resource_shape}")
            affinity = LeptonResourceAffinity(allowed_dedicated_node_groups=[dedicated_node_group])
        else:
            affinity = None
        envs = [EnvVar(name='CLOUD_METADATA', value=json.dumps(cloud_metadata))]
        if num_nodes > 1:
            envs += [EnvVar(name="WORLD_SIZE", value=str(num_nodes))]
            envs += [EnvVar(name="SAVE_ON_EACH_NODE", value="True")]

        lepton_job = LeptonJob(
            metadata=Metadata(
                name=self.get_lepton_job_name(job_id),
                visibility=LeptonVisibility.PRIVATE
            ),
            spec=LeptonJobUserSpec(
                container=LeptonContainer(
                    image=image,
                    command=["/bin/bash", "-c", command],
                ),
                affinity=affinity,
                intra_job_communication=True if num_nodes > 1 else None,
                parallelism=num_nodes if num_nodes > 1 else None,
                completions=num_nodes,
                resource_shape=resource_shape,
                image_pull_secrets=image_pull_secrets,
                log=LeptonLog(enable_collection=True,
                              save_termination_logs=os.getenv("DEBUG_MODE", "false").lower() == "true"),
                envs=envs
            )
        )
        self.api_client.job.create(lepton_job)
        logger.info(f"Lepton Job {job_id} created successfully")
        return lepton_job

    def send_deployment_request(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        max_retries=3,
        retry_delay=5,
    ):
        """Send a request to the Lepton microservice with retry logic."""
        for attempt in range(max_retries):
            try:
                deployment = self.wait_for_deployment(job_id)
                if deployment:
                    base_url = deployment.status.endpoint.external_endpoint
                    if base_url:
                        return send_microservice_request(
                            base_url,
                            api_endpoint,
                            network,
                            action,
                            cloud_metadata,
                            specs,
                            job_id,
                            docker_env_vars,
                            cloud_based=False)
                    logger.error(f"Deployment {job_id} has no external endpoint (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                logger.error(f"Timed out waiting for deployment {job_id} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            except Exception as e:
                logger.error(f"Error in send_deployment_request for {job_id} "
                             f"(attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logger.error(f"All retry attempts failed for deployment {job_id}")
                return None
        return None
