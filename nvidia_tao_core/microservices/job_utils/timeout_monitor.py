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

"""Job timeout monitoring and termination utilities"""
import os
import logging
from datetime import datetime, timezone

from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    get_dnn_status,
    get_handler_job_metadata,
    update_job_status,
    save_automl_controller_info,
    get_automl_controller_info,
    internal_job_status_update,
    get_health_beat
)
from nvidia_tao_core.microservices.constants import JOB_STATUS_TIMEOUT_MINUTES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_last_status_timestamp(job_id, automl=False, experiment_number="0"):
    """Get the timestamp of the last status update for a job"""
    try:
        status_data = get_dnn_status(job_id, automl=automl, experiment_number=experiment_number)
        if not status_data:
            logger.info(f"No status data found for job {job_id}")
            return None

        # Find the most recent timestamp in the status data
        latest_timestamp = None
        for status_entry in status_data:
            if isinstance(status_entry, dict) and 'timestamp' in status_entry:
                try:
                    timestamp_str = status_entry['timestamp']
                    if isinstance(timestamp_str, str):
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%f']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format matches, try parsing as ISO format
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        continue

                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                except (ValueError, TypeError) as e:
                    logger.info(f"Failed to parse timestamp {status_entry.get('timestamp')} for job {job_id}: {e}")
                    continue

        return latest_timestamp

    except Exception as e:
        logger.error(f"Error getting last status timestamp for job {job_id}: {e}")
        return None


def check_pod_liveness(job_id):
    """Check if a pod/container is alive by hitting its liveness endpoint

    Supports both Kubernetes and Docker Compose backends.

    Returns:
        bool: True if pod/container is alive and responding, False otherwise
    """
    try:
        import requests
        from nvidia_tao_core.microservices.handlers.stateless_handlers import BACKEND

        port = 8000

        if BACKEND == "local-docker":
            # Docker Compose: use container name directly
            # Container name is the same as job_id
            liveness_url = f"http://{job_id}:{port}/api/v1/health/liveness"
            logger.info(f"Checking liveness for Docker container {job_id} at {liveness_url}")
        else:
            # Kubernetes: use service name with namespace
            from nvidia_tao_core.microservices.handlers.utilities import get_statefulset_service_name
            service_name = get_statefulset_service_name(job_id)
            namespace = os.getenv('NAMESPACE', 'default')

            # Format: http://service-name.namespace.svc.cluster.local:8000/api/v1/health/liveness
            liveness_url = f"http://{service_name}.{namespace}.svc.cluster.local:{port}/api/v1/health/liveness"
            logger.info(f"Checking liveness for K8s job {job_id} at {liveness_url}")

        # Make request with short timeout
        response = requests.get(liveness_url, timeout=5)

        if response.status_code == 200:
            logger.info(f"Job {job_id} is alive and responding to liveness checks (backend: {BACKEND})")
            return True

        logger.info(f"Job {job_id} liveness check returned status {response.status_code}")
        return False

    except requests.exceptions.RequestException as e:
        logger.info(f"Job {job_id} liveness check failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking liveness for job {job_id}: {e}")
        return False


def check_brain_job_timeout(job_info):
    """Check if an AutoML brain job has timed out based on health beats

    Brain jobs are K8s jobs (not StatefulSets), so they don't have liveness endpoints.
    Instead, we rely on health beats reported by the controller during its execution loop.

    Args:
        job_info: Dictionary containing job information including job_id

    Returns:
        bool: True if brain job has timed out, False otherwise
    """
    job_id = job_info.get('job_id')

    if not job_id:
        return False

    try:
        timeout_seconds = JOB_STATUS_TIMEOUT_MINUTES * 60

        # Get the last health beat timestamp
        health_beat = get_health_beat(job_id)

        if health_beat is None:
            # No health beat found - brain might not have started yet
            # Check job creation time
            job_metadata = get_handler_job_metadata(job_id)
            if job_metadata:
                created_on = job_metadata.get("created_on")
                last_modified = job_metadata.get("last_modified")

                job_start_time = None
                for timestamp_field in [last_modified, created_on]:
                    if timestamp_field:
                        try:
                            if isinstance(timestamp_field, str):
                                job_start_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                            elif isinstance(timestamp_field, datetime):
                                job_start_time = timestamp_field
                            break
                        except (ValueError, TypeError):
                            continue

                if job_start_time:
                    current_time = datetime.now(tz=timezone.utc)
                    time_since_start = current_time - job_start_time

                    if time_since_start.total_seconds() > timeout_seconds:
                        timeout_message = (
                            f"AutoML brain job timed out: no health beats received for "
                            f"{time_since_start.total_seconds():.0f}s (timeout: {timeout_seconds}s). "
                            "Controller may have failed to start. Terminating."
                        )
                        logger.warning(f"Brain job {job_id}: {timeout_message}")

                        # Update job status
                        internal_job_status_update(
                            job_id=job_id,
                            automl=False,
                            message=timeout_message,
                            status="FAILURE",
                            handler_id=job_info.get('handler_id'),
                            kind=job_info.get('kind')
                        )
                        return True

                    logger.info(
                        f"Brain job {job_id} has no health beats yet but started only "
                        f"{time_since_start.total_seconds():.0f}s ago. Giving it time to initialize."
                    )
                    return False

            logger.info(f"Brain job {job_id} has no health beats and cannot determine start time. Being conservative.")
            return False

        # We have a health beat - check if it's recent
        last_beat_time = health_beat.get('last_beat')
        if last_beat_time:
            if isinstance(last_beat_time, str):
                last_beat_time = datetime.fromisoformat(last_beat_time.replace('Z', '+00:00'))

            current_time = datetime.now(tz=timezone.utc)
            time_since_beat = current_time - last_beat_time

            if time_since_beat.total_seconds() > timeout_seconds:
                timeout_message = (
                    f"AutoML brain job timed out: last health beat was {time_since_beat.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s). Controller appears to be stuck. Terminating."
                )
                logger.warning(f"Brain job {job_id}: {timeout_message}")

                # Update job status
                internal_job_status_update(
                    job_id=job_id,
                    automl=False,
                    message=timeout_message,
                    status="FAILURE",
                    handler_id=job_info.get('handler_id'),
                    kind=job_info.get('kind')
                )
                return True

            logger.info(
                f"Brain job {job_id} last health beat: {time_since_beat.total_seconds():.0f}s ago "
                f"(timeout: {timeout_seconds}s). Message: {health_beat.get('message', 'N/A')}"
            )
            return False

        return False

    except Exception as e:
        logger.error(f"Error checking brain job timeout for {job_id}: {e}")
        return False


def check_job_timeout(job_info):
    """Check if a job has timed out based on last status update

    For AutoML brain jobs, uses health beats instead of pod liveness checks
    since brain jobs are K8s jobs (not StatefulSets with services).
    """
    job_id = job_info.get('job_id')
    is_automl = job_info.get('is_automl', False)
    is_automl_brain = job_info.get('is_automl_brain', False)
    experiment_number = job_info.get('experiment_number', '0')
    brain_job_id = job_info.get('brain_job_id', None)

    if not job_id:
        return False

    try:
        # Special handling for AutoML brain jobs - use health beats
        if is_automl_brain:
            return check_brain_job_timeout(job_info)

        timeout_seconds = JOB_STATUS_TIMEOUT_MINUTES * 60
        last_timestamp = get_last_status_timestamp(job_id, automl=is_automl, experiment_number=experiment_number)

        job_description = f"AutoML experiment {experiment_number} for job {job_id}" if is_automl else f"Job {job_id}"

        # CASE 1: We have status updates - check if they're recent
        if last_timestamp is not None:
            current_time = datetime.now(tz=timezone.utc)
            time_since_update = current_time - last_timestamp
            is_timed_out = time_since_update.total_seconds() > timeout_seconds

            if is_timed_out:
                timeout_message = (
                    f"Job timed out: last status update was {time_since_update.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s). Terminating due to inactivity."
                )
                logger.warning(f"{job_description} {timeout_message}")

                # Update job status before terminating
                internal_job_status_update(
                    job_id=job_id,
                    automl=is_automl,
                    automl_experiment_number=experiment_number,
                    message=timeout_message,
                    status="FAILURE",
                    handler_id=job_info.get('handler_id'),
                    kind=job_info.get('kind')
                )
            else:
                logger.info(
                    f"{job_description} last status update: {time_since_update.total_seconds():.0f}s ago "
                    f"(timeout: {timeout_seconds}s)"
                )

            return is_timed_out

        # CASE 2: No status updates - check if pod is alive via liveness endpoint
        logger.info(
            f"No status updates found for {job_description}. "
            "Checking if pod is alive via liveness endpoint..."
        )

        # Check if the pod is responding to liveness checks
        # Use actual_job_id which includes experiment_number for AutoML
        pod_is_alive = check_pod_liveness(job_id)

        if pod_is_alive:
            # Pod is running and responding to liveness but NOT sending status updates
            # Check how long the job has been in this state before timing out
            logger.info(
                f"{job_description} pod is alive (liveness check passed) but has sent no status updates. "
                "Checking job age to determine if timeout applies..."
            )

            # Get job creation/start time from job metadata
            lookup_job_id = job_id if not brain_job_id else brain_job_id
            job_metadata = get_handler_job_metadata(lookup_job_id)
            if job_metadata:
                last_modified = job_metadata.get("last_modified")
                created_on = job_metadata.get("created_on")

                # Use the most recent timestamp
                job_start_time = None
                for timestamp_field in [last_modified, created_on]:
                    if timestamp_field:
                        try:
                            if isinstance(timestamp_field, str):
                                job_start_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                            elif isinstance(timestamp_field, datetime):
                                job_start_time = timestamp_field
                            break
                        except (ValueError, TypeError):
                            continue

                if job_start_time:
                    current_time = datetime.now(tz=timezone.utc)
                    time_since_start = current_time - job_start_time

                    if time_since_start.total_seconds() > timeout_seconds:
                        timeout_message = (
                            f"Job timed out: pod is alive but has sent no status updates for "
                            f"{time_since_start.total_seconds():.0f}s (timeout: {timeout_seconds}s). "
                            "Job may be stuck or failing to report progress. Terminating."
                        )
                        logger.warning(f"{job_description} {timeout_message}")

                        # Update job status before terminating
                        internal_job_status_update(
                            job_id=lookup_job_id,
                            automl=is_automl,
                            automl_experiment_number=experiment_number,
                            message=timeout_message,
                            status="FAILURE",
                            handler_id=job_info.get('handler_id'),
                            kind=job_info.get('kind')
                        )
                        return True  # Timeout this job

                    logger.info(
                        f"{job_description} pod is alive but no status updates yet. "
                        f"Job started {time_since_start.total_seconds():.0f}s ago. "
                        f"Giving it more time (timeout: {timeout_seconds}s)."
                    )
                    return False  # Still within grace period

            # If we can't determine job start time, don't timeout (be conservative)
            logger.info(
                f"{job_description} pod is alive but no status updates and cannot determine job age. "
                "Not applying timeout to be conservative."
            )
            return False

        # Pod is not responding yet - likely still waiting for resources or starting up
        logger.info(
            f"{job_description} pod is not responding to liveness checks yet. "
            "Likely waiting for GPU resources or still initializing. No timeout applied."
        )
        return False  # Don't timeout - job is still initializing

    except Exception as e:
        logger.error(f"Error checking timeout for job {job_id} (AutoML: {is_automl}): {e}")
        return False


def terminate_timed_out_job(job_info):
    """Terminate a timed out job"""
    from nvidia_tao_core.microservices.job_utils.executor.statefulset_executor import StatefulSetExecutor
    from nvidia_tao_core.microservices.job_utils.executor.job_executor import JobExecutor

    job_id = job_info.get('job_id')
    handler_id = job_info.get('handler_id')
    kind = job_info.get('kind', '')
    is_automl = job_info.get('is_automl', False)
    is_automl_brain = job_info.get('is_automl_brain', False)
    brain_job_id = job_info.get('brain_job_id', None)
    experiment_number = job_info.get('experiment_number', '0')

    if not job_id or not handler_id:
        logger.error(f"Cannot terminate job: missing job_id or handler_id in {job_info}")
        return False

    try:
        # Handle AutoML brain jobs differently - they're K8s jobs, not StatefulSets
        if is_automl_brain:
            logger.info(f"Terminating timed out AutoML brain job {job_id}")

            # Update job status to indicate timeout
            update_job_status(handler_id, job_id, status="Error", kind=kind)

            # Delete the K8s Job (not StatefulSet)
            job_executor = JobExecutor()
            job_executor.delete_job(job_id)

            logger.info(f"Deletion request sent for timed out AutoML brain job {job_id}")
            return True

        if is_automl and brain_job_id:
            logger.info(f"Terminating timed out AutoML experiment {experiment_number} for job {brain_job_id}")

            # For AutoML experiments, we need to update the specific experiment status
            # and potentially terminate the StatefulSet for that experiment
            controller_info = get_automl_controller_info(brain_job_id)
            logger.info(f"Controller info for job {brain_job_id}: {controller_info}")

            if isinstance(controller_info, list):
                updated_controller = []
                experiment_found = False

                for recommendation in controller_info:
                    if isinstance(recommendation, dict):
                        if str(recommendation.get("id", "")) == experiment_number:
                            # Mark this experiment as failed due to timeout
                            recommendation["status"] = "failure"
                            recommendation["message"] = "Terminated due to timeout - no status updates received"
                            experiment_found = True
                            logger.info(f"Marked AutoML experiment {experiment_number} as error due to timeout")
                        updated_controller.append(recommendation)

                if experiment_found:
                    # Save the updated controller info
                    logger.info(
                        f"Saving updated controller info for AutoML experiment {experiment_number} "
                        f"for job {job_id}"
                    )
                    logger.info(f"Updated controller info: {updated_controller}")
                    logger.info(f"Brain job id: {brain_job_id}")
                    save_automl_controller_info(brain_job_id, updated_controller)
                    get_automl_controller_info(brain_job_id)

                    # Try to terminate the StatefulSet for this specific experiment
                    # The StatefulSet name for AutoML experiments typically includes the experiment number
                    statefulset_executor = StatefulSetExecutor()
                    success = statefulset_executor.delete_statefulset(job_id, use_ngc=True)

                    if success:
                        logger.info(
                            f"Successfully terminated timed out AutoML experiment {experiment_number} for job {job_id}"
                        )
                    else:
                        logger.error(
                            f"Failed to terminate StatefulSet for AutoML experiment {experiment_number} "
                            f"for job {job_id}"
                        )

                    return success
                logger.warning(f"AutoML experiment {experiment_number} not found in controller info for job {job_id}")
                return False
            logger.error(f"Invalid controller info format for AutoML job {job_id}")
            return False
        logger.info(f"Terminating timed out job {job_id}")

        # Update job status to indicate timeout
        update_job_status(handler_id, job_id, status="Error", kind=kind)

        # Delete the StatefulSet
        statefulset_executor = StatefulSetExecutor()
        success = statefulset_executor.delete_statefulset(job_id, use_ngc=True)

        if success:
            logger.info(f"Successfully terminated timed out job {job_id}")
        else:
            logger.error(f"Failed to terminate timed out job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error terminating timed out job {job_id} (AutoML: {is_automl}): {e}")
        return False
