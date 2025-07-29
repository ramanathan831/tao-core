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

"""Inference Microservice handler using StatefulSets for long-lived inference"""
import logging
import requests
from datetime import datetime
from typing import Dict, Any
import os

from nvidia_tao_core.microservices.handlers.docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.handlers.utilities import Code, add_workspace_to_cloud_metadata
from nvidia_tao_core.microservices.job_utils import executor as jobDriver
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_handler_metadata


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceMicroserviceHandler:
    """Handler class for long-lived Inference Microservice jobs using StatefulSets

    This handler manages:
    - Starting long-lived Inference Microservice using StatefulSets for persistence
    - Running both HTTP server (for file operations) and container_handler.py (for TAO jobs)
    - Managing service lifecycle (start, update, stop)
    - Following standard TAO container and job processing patterns
    """

    @staticmethod
    def start_inference_microservice(org_name: str, experiment_id: str, job_id: str,
                                     job_config: Dict[str, Any],
                                     replicas: int = 1) -> Code:
        """Starts a long-lived Inference Microservice using StatefulSet"""
        logger.info("Starting Inference Microservice %s for experiment %s", job_id, experiment_id)

        # Get the Inference Microservice Docker image
        image = DOCKER_IMAGE_MAPPER.get("VILA", "nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf2.11.0")
        logger.info("Using Docker image: %s", image)

        # StatefulSet name
        statefulset_name = f"ims-{job_id}"
        logger.info("Using StatefulSet name: %s", statefulset_name)

        # Build command for Inference Microservice integrated into TAO container
        model_path = job_config.get("model_path", "")
        logger.info("Using model path: %s", model_path)
        if not model_path:
            return Code(400, {}, "Model path is required for Inference Microservice")

        # cli_args = convert_dict_to_cli_args(job_config)
        # cli_args = " ".join(cli_args)
        # logger.info("Using CLI args: %s", cli_args)

        experiment_metadata = get_handler_metadata(experiment_id, kind="experiments")
        docker_env_vars = experiment_metadata.get("docker_env_vars", {})
        workspace_id = experiment_metadata.get("workspace", "")
        workspace_metadata = get_handler_metadata(workspace_id, kind="workspaces")
        cloud_metadata = {}
        add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata)
        logger.info("workspace_metadata %s", workspace_metadata)
        cloud_type = workspace_metadata.get('cloud_type', '')
        cloud_details = workspace_metadata.get('cloud_specific_details', {})
        bucket_name = cloud_details.get('cloud_bucket_name', '')
        job_config["results_dir"] = f"{cloud_type}://{bucket_name}/results/inference_microservice_results"

        job_metadata = {
            "job_id": job_id,
            "specs": job_config,
            "cloud_metadata": cloud_metadata,
            "neural_network_name": "vila",
        }

        # Clean TAO-compliant StatefulSet setup: Pure container_handler.py approach
        run_command = f"""
# Start clean TAO Inference Microservice StatefulSet container (no embedded HTTP server)
umask 0 &&
exec python3 -m llava.cli.tao_model_server --job "{str(job_metadata)}" --docker_env_vars "{str(docker_env_vars)}"
        """
        logger.info("Using run command: %s", run_command)

        # Ports for Inference Microservice (HTTP API, health check)
        ports = (8080, 8081)

        try:
            # Create long-lived inference service StatefulSet
            # IMPORTANT: This overrides the default container entrypoint (e.g., "flask run")
            # with our custom command that starts the persistent model server + container_handler.py
            success = jobDriver.create_inference_microservice_statefulset(
                statefulset_name=statefulset_name,
                image=image,
                command=run_command,  # This overrides default entrypoint with bash -c "run_command"
                replicas=replicas,
                num_gpu=1,
                ports=ports,
                is_long_lived=True,
                org_name=org_name,
                experiment_id=experiment_id,
                job_id=job_id
            )

            if not success:
                return Code(500, {}, "Failed to create Inference Microservice StatefulSet")

            logger.info("Inference Microservice %s is ready", statefulset_name)

            # Create K8s service for the StatefulSet
            service_id = f"ims-svc-{job_id}"
            return InferenceMicroserviceHandler._create_inference_microservice_service(
                service_id=service_id,
                statefulset_name=statefulset_name,
                ports=ports,
                job_id=job_id
            )

        except Exception as e:
            logger.error("Error starting Inference Microservice: %s", str(e))
            return Code(500, {}, f"Failed to start Inference Microservice: {str(e)}")

    @staticmethod
    def _create_inference_microservice_service(service_id: str, statefulset_name: str, ports: tuple,
                                               job_id: str) -> Code:
        """Create Kubernetes service for Inference Microservice StatefulSet"""
        try:
            # Create service for external access
            service_info = jobDriver.create_inference_microservice_service(
                service_name=service_id,
                statefulset_name=statefulset_name,
                ports=ports,
                service_type="ClusterIP"
            )

            # Get service endpoint information
            service_url = f"http://{service_info.get('cluster_ip', 'localhost')}:{ports[0]}"

            logger.info("Inference Microservice created at %s", service_url)

            return Code(200, {
                "service_id": service_id,
                "service_url": service_url,
                "status": "Running",
                "endpoints": {
                    "inference": f"{service_url}/inference",
                    "health": f"{service_url}/health",
                    "status": f"{service_url}/status"
                },
                "job_id": job_id
            }, "Inference Microservice started successfully")

        except Exception as e:
            logger.error("Error creating Inference Microservice: %s", str(e))
            return Code(500, {}, f"Failed to create Inference Microservice: {str(e)}")

    @staticmethod
    def stop_inference_microservice(job_id: str, auto_deletion: bool = False) -> Code:
        """Stop a running Inference Microservice

        Args:
            job_id: Job ID for the microservice to stop
            auto_deletion: True if called due to idle timeout, False if manual stop

        Returns:
            Code object with status and result information
        """
        action = "Auto-deleting" if auto_deletion else "Stopping"
        reason = "due to inactivity" if auto_deletion else "manually"

        logger.info("%s Inference Microservice %s %s", action, job_id, reason)

        statefulset_name = f"ims-{job_id}"
        service_name = statefulset_name.replace("ims-", "ims-svc-")
        logger.info("Using StatefulSet name: %s", statefulset_name)
        logger.info("Using Service name: %s", service_name)

        try:
            # Delete the StatefulSet and associated service directly
            statefulset_deleted = jobDriver.delete_inference_microservice_statefulset(statefulset_name)
            service_deleted = jobDriver.delete_inference_microservice_service(service_name)

            if statefulset_deleted and service_deleted:
                success_message = "auto-deleted due to inactivity" if auto_deletion else "stopped successfully"
                logger.info(
                    "Successfully %s Inference Microservice %s",
                    "auto-deleted" if auto_deletion else "stopped", job_id
                )

                result = {
                    "status": "success",
                    "message": f"Inference microservice for job {job_id} {success_message}",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "auto_deletion": auto_deletion
                }
                return Code(200, result, f"Inference Microservice {success_message}")
            error_details = []
            if not statefulset_deleted:
                error_details.append("StatefulSet deletion failed")
            if not service_deleted:
                error_details.append("Service deletion failed")

            error_msg = (
                f"Failed to {action.lower()} inference microservice for job {job_id}: "
                f"{', '.join(error_details)}"
            )
            logger.error(error_msg)
            return Code(500, {"error": error_msg}, f"Failed to {action.lower()} Inference Microservice")

        except Exception as e:
            logger.error("Error %s Inference Microservice %s: %s", action.lower(), job_id, str(e))
            return Code(500, {"error": str(e)}, f"Error {action.lower()} Inference Microservice")

    @staticmethod
    def get_inference_microservice_status(job_id: str) -> Code:
        """Gets the status of a Inference Microservice StatefulSet"""
        statefulset_name = f"ims-{job_id}"

        try:
            stat_dict = jobDriver.status_inference_microservice_statefulset(statefulset_name)
            status = stat_dict.get("status", "Unknown")

            return Code(200, {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": status,
                "replicas": stat_dict.get("replicas", {}),
                "pods": []
            }, f"Inference Microservice status: {status}")

        except Exception as e:
            logger.error("Error getting Inference Microservice status: %s", str(e))
            return Code(500, {}, f"Failed to get service status: {str(e)}")

    @staticmethod
    def check_inference_microservice_model_readiness(job_id: str) -> dict:
        """Check if Inference Microservice model is ready in StatefulSet containers"""
        try:
            statefulset_name = f"ims-{job_id}"

            # Check if StatefulSet pods exist and are running
            try:
                stat_dict = jobDriver.status_inference_microservice_statefulset(statefulset_name)
                statefulset_status = stat_dict.get("status", "Unknown")

                if statefulset_status == "Running":
                    # If StatefulSet is running, model should be loaded (loaded at startup)
                    return {
                        "job_id": job_id,
                        "status": "ready",
                        "loaded": True,
                        "statefulset_status": statefulset_status
                    }
                return {
                    "job_id": job_id,
                    "status": "not_ready",
                    "loaded": False,
                    "statefulset_status": statefulset_status
                }
            except Exception:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "loaded": False,
                    "statefulset_status": "NotFound"
                }

        except Exception as e:
            logger.error(f"Error checking Inference Microservice model readiness: {e}")
            return {"status": "error", "error": str(e), "loaded": False}

    @staticmethod
    def get_inference_microservice_url(job_id: str, endpoint: str = "inference") -> str:
        """Get the URL for inference microservice requests

        Args:
            job_id: Job ID for the microservice
            endpoint: Endpoint to call (inference, health, status)
            use_fqdn: Deprecated parameter, kept for compatibility

        Returns:
            Full URL for the request (always uses simple service name)
        """
        service_name = f"ims-svc-{job_id}"

        # Always use simple service name for both Kubernetes and docker-compose
        # This works reliably for intra-cluster communication and avoids DNS issues
        if os.environ.get('BACKEND', 'local-k8s') == 'local-docker':
            url = f"http://{job_id}:8080/{endpoint}"
        else:
            url = f"http://{service_name}:8080/{endpoint}"

        logger.info(f"Inference microservice URL: {url}")
        return url

    @staticmethod
    def process_inference_microservice_request_direct(job_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request directly to the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice
            request_data: Request data to send to the microservice

        Returns:
            Response from the microservice
        """
        try:
            logger.info(f"Processing inference request for job {job_id}")

            # Get the inference URL
            inference_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "inference"
            )

            # Make request to the microservice
            timeout = 300  # 5 minutes timeout for inference
            response = requests.post(
                inference_url,
                json=request_data,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Inference request completed successfully for job {job_id}")
                return result
            if response.status_code in [202, 503]:
                # Server is initializing or loading - return appropriate response
                result = response.json()
                logger.info(
                    f"Inference microservice not ready for job {job_id}: "
                    f"{result.get('message', 'Unknown status')}"
                )
                return result
            error_msg = f"Inference request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            try:
                error_detail = response.json()
                error_msg += f": {error_detail.get('error', 'Unknown error')}"
            except (ValueError, KeyError):
                error_msg += f": {response.text}"

            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

        except requests.exceptions.Timeout:
            error_msg = f"Inference request timed out after {timeout} seconds for job {job_id}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to inference microservice for job {job_id}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Unexpected error during inference request for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

    @staticmethod
    def get_inference_microservice_status_direct(job_id: str) -> Dict[str, Any]:
        """Get status directly from the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice

        Returns:
            Status response from the microservice
        """
        try:
            logger.info(f"Getting status for inference microservice job {job_id}")

            # Get the status URL
            status_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "status"
            )

            # Make request to the microservice
            response = requests.get(status_url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Status retrieved successfully for job {job_id}")
                return result
            error_msg = f"Status request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Failed to get status for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

    @staticmethod
    def get_inference_microservice_status_detailed(job_id: str) -> dict:
        """Get Inference Microservice service status with model readiness information"""
        try:
            statefulset_name = f"ims-{job_id}"
            stat_dict = jobDriver.status_inference_microservice_statefulset(statefulset_name)

            # Check model readiness
            model_state = InferenceMicroserviceHandler.check_inference_microservice_model_readiness(job_id)

            return {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": stat_dict.get("status", "Unknown"),
                "replicas": stat_dict.get("replicas", {}),
                "model_loaded": model_state.get("loaded", False),
                "model_status": model_state.get("status", "unknown")
            }

        except Exception as e:
            logger.error(f"Error getting Inference Microservice service status: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
