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

"""Handler to execute jobs on NVIDIA Cloud Functions (NVCF)"""

import logging
import traceback
from nvidia_tao_core.microservices.handlers.execution_handlers.execution_handler import ExecutionHandler
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_job_metadata,
    write_job_metadata,
    internal_job_status_update,
    update_job_message,
    get_handler_metadata
)
from nvidia_tao_core.microservices.utils.nvcf_utils import (
    create_function,
    deploy_function,
    get_available_nvcf_instances,
    get_function,
    delete_function_version,
    add_authorized_party,
    create_microservice_job_on_nvcf,
    get_nvcf_microservices_job_status
)
from nvidia_tao_core.microservices.utils.executor_utils import override_k8_status
from nvidia_tao_core.microservices.enum_constants import Backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_nvcf_handler_from_workspace(workspace_id):
    """Get the NVCF handler from the workspace metadata

    Args:
        workspace_id: Workspace identifier

    Returns:
        NvcfHandler instance if NVCF credentials found, None otherwise
    """
    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})
    nvcf_backend_details = cloud_specific_details.get("nvcf_backend_details")
    team_name = cloud_specific_details.get("teamName")
    ngc_key = cloud_specific_details.get("ngc_key")
    org_name = workspace_metadata.get("org_name")

    if nvcf_backend_details and team_name and ngc_key and org_name:
        try:
            logger.info(f"Instantiating NVCF Handler for workspace {workspace_id}")
            return NvcfHandler(org_name, team_name, ngc_key, nvcf_backend_details)
        except Exception as e:
            logger.error(f"Exception instantiating NVCF Handler: {e}")
    return None


class NvcfHandler(ExecutionHandler):
    """Handler to execute jobs on NVIDIA Cloud Functions (NVCF)"""

    def __init__(self, org_name, team_name, ngc_key, nvcf_backend_details):
        """Initialize the NVCF Handler

        Args:
            org_name: Organization name
            team_name: NVCF team name
            ngc_key: NGC API key for authentication
            nvcf_backend_details: Dictionary containing NVCF backend configuration
                (cluster, gpu_type, instance_type, etc.)
        """
        super().__init__(backend_type=Backend.NVCF)
        self.org_name = org_name
        self.team_name = team_name
        self.ngc_key = ngc_key
        self.nvcf_backend_details = nvcf_backend_details
        self.logger.info(f"NvcfHandler initialized for org {org_name}, team {team_name}")

    def get_available_instances(self, **kwargs):
        """Get the available instances for the NVCF backend"""
        user_id = kwargs.get("user_id", None)
        if not user_id:
            raise ValueError("user_id is required")
        available_nvcf_instances = get_available_nvcf_instances(user_id, self.org_name)
        return available_nvcf_instances

    def create_job(self, job_name, image, docker_image_name, nv_job_metadata,
                   num_nodes=1, automl_exp_job=False):
        """Creates an NVCF function and deploys it

        Args:
            job_name: Job identifier
            image: Container image to use
            docker_image_name: Docker image name for NVCF
            nv_job_metadata: Job metadata dictionary
            num_nodes: Number of nodes to deploy
            automl_exp_job: Whether this is an AutoML experiment job

        Returns:
            None (updates job metadata in place)
        """
        deployment_string = nv_job_metadata.get("deployment_string", "")
        current_available = self.nvcf_backend_details.get("current_available", 1)
        num_nodes = min(num_nodes, current_available)

        if not deployment_string:
            # Create NVCF function
            create_response = create_function(
                self.org_name,
                self.team_name,
                job_name,
                docker_image_name,
                self.ngc_key
            )

            if create_response.ok:
                self.logger.info(f"Function created successfully for job {job_name}")
                function_metadata = create_response.json()
                function_id = function_metadata["function"]["id"]
                version_id = function_metadata["function"]["versionId"]

                # Deploy NVCF function
                deploy_response = deploy_function(
                    self.org_name,
                    self.team_name,
                    function_metadata,
                    self.nvcf_backend_details,
                    self.ngc_key,
                    image=docker_image_name,
                    num_nodes=num_nodes
                )

                if deploy_response.ok:
                    deployment_string = f"{function_id}:{version_id}"
                    self.logger.info(f"Function deployment initiated successfully for job {job_name}")
                else:
                    internal_job_status_update(
                        job_name,
                        automl=automl_exp_job,
                        automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                        message="NVCF function could not be deployed"
                    )
                    self.logger.error(f"Function deployment request failed for job {job_name}")
                    self.logger.error(f"Deployment response {deploy_response.text}")
                    raise ValueError(f"Function deployment request failed for job {job_name}")
            else:
                internal_job_status_update(
                    job_name,
                    automl=automl_exp_job,
                    automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                    message="NVCF function couldn't be created, retry job again"
                )
                self.logger.error(f"Function creation request failed for job {job_name}")
                raise ValueError("NVCF function couldn't be created, retry job again")

        # Update job metadata with deployment string
        job_metadata = get_handler_job_metadata(job_name)
        job_metadata["backend_details"] = {}
        nv_job_metadata["deployment_string"] = deployment_string
        job_metadata["backend_details"]["nvcf_metadata"] = nv_job_metadata
        write_job_metadata(job_name, job_metadata)

    def delete(self, job_id):
        """Deletes an NVCF Function

        Args:
            job_name: Job identifier
        """
        job_metadata = get_handler_job_metadata(job_id)
        nv_job_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
        ngc_key = nv_job_metadata.get("TAO_USER_KEY", self.ngc_key)
        deployment_string = nv_job_metadata.get("deployment_string", "")

        if deployment_string.find(":") == -1:
            self.logger.warning(f"Deployment not active yet {job_id}")
            return

        function_id, version_id = deployment_string.split(":")
        delete_function_version(
            self.org_name,
            self.team_name,
            function_id,
            version_id,
            ngc_key
        )
        self.logger.info(f"NVCF function deleted for job {job_id}")

    def get_job_status(self, job_id, **kwargs):
        """Returns status of NVCF job

        Args:
            job_id: Job identifier
            **kwargs: Additional parameters

        Returns:
            str: Job status (Pending, Running, Done, Error, etc.)
        """
        handler_kind = kwargs.get("handler_kind", "")
        docker_env_vars = kwargs.get("docker_env_vars", {})
        automl_exp_job = kwargs.get("automl_exp_job", False)
        authorized_party_nca_id = kwargs.get("authorized_party_nca_id", "")
        try:
            job_metadata = get_handler_job_metadata(job_id)
            job_handler_id = job_metadata.get("handler_id", "")
            nv_job_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
            job_status = job_metadata.get("status", "Pending")
            team_name = nv_job_metadata.get("teamName", self.team_name)
            ngc_key = docker_env_vars.get("TAO_USER_KEY", self.ngc_key)
            deployment_string = nv_job_metadata.get("deployment_string", "")
            tao_api_status_callback_url = docker_env_vars.get("TAO_LOGGING_SERVER_URL", "")
            job_message_job_id = tao_api_status_callback_url.split("/")[-1] if tao_api_status_callback_url else job_id

            if job_status == "Pending":
                if deployment_string.find(":") != -1:
                    function_id, version_id = deployment_string.split(":")
                    nvcf_function_response = get_function(
                        self.org_name,
                        team_name,
                        function_id,
                        version_id,
                        ngc_key
                    )

                    if nvcf_function_response.status_code == 200:
                        nvcf_function_metadata = nvcf_function_response.json()
                        update_job_message(
                            job_handler_id,
                            job_message_job_id,
                            handler_kind,
                            "NVCF function is being deployed",
                            automl_expt_job_id=job_id,
                            update_automl_expt=True
                        )
                    else:
                        internal_job_status_update(
                            job_id,
                            automl=automl_exp_job,
                            automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                            message="NVCF function details cant be retrieved"
                        )
                        return "Error"

                    if nvcf_function_metadata.get("function", {}).get("status") == "ACTIVE":
                        self.logger.info("NVCF function is active, creating microservice job on NVCF")
                        deployment_string = (
                            f"{nvcf_function_metadata['function']['id']}:"
                            f"{nvcf_function_metadata['function']['versionId']}"
                        )
                        job_status, message = create_microservice_job_on_nvcf(
                            job_metadata, docker_env_vars=docker_env_vars
                        )
                        job_metadata["status"] = job_status
                        if job_metadata.get("job_details", {}).get(job_id, {}):
                            job_metadata["job_details"][job_id]["detailed_status"]["message"] = message
                        write_job_metadata(job_id, job_metadata)

                        if authorized_party_nca_id:
                            self.logger.info(
                                f"Adding authorized party {authorized_party_nca_id} for job {job_id}")
                            add_authorized_party(
                                self.org_name,
                                team_name,
                                function_id,
                                version_id,
                                authorized_party_nca_id,
                                ngc_key
                            )

                    if nvcf_function_metadata.get("function", {}).get("status") == "ERROR":
                        self.logger.error(f"Get function deployment status for job {job_id} returned error")
                        internal_job_status_update(
                            job_id,
                            automl=automl_exp_job,
                            automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                            message="NVCF function metadata has ERROR status"
                        )
                        return "Error"

            # Check for toolkit status override
            override_status = override_k8_status(job_id, job_status)
            job_status = get_nvcf_microservices_job_status(
                job_metadata,
                status=override_status,
                docker_env_vars=docker_env_vars
            )

            if override_status and override_status != job_status:
                job_status = override_status
                self.logger.warning(
                    f"job metadata status is {job_status}, Toolkit Status is {override_status}, so overwriting")
                self.logger.warning(f"Microservices job status via NVCF is {job_status}")

            return job_status

        except Exception as e:
            self.logger.error(f"Exception caught for {job_id} {e}")
            self.logger.error(traceback.format_exc())
            return "Error"
