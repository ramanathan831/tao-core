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

"""Deployment executor for Triton and Tensorboard deployments"""
import os
import requests
from kubernetes import client
from kubernetes.client.rest import ApiException

from nvidia_tao_core.microservices.job_utils.executor.base_executor import BaseExecutor

if os.getenv("BACKEND"):  # To see if the container is going to be used for Service pods or network jobs
    from nvidia_tao_core.microservices.handlers.mongo_handler import (
        mongo_secret,
        mongo_operator_enabled,
        mongo_namespace
    )


class DeploymentExecutor(BaseExecutor):
    """Handles Kubernetes Deployment operations for Triton and Tensorboard"""

    def create_triton_deployment(self, deployment_name, image, command, replicas, num_gpu=-1, ports=(8000, 8001, 8002)):
        """Creates a Triton deployment"""
        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        dshm_volume_mount = client.V1VolumeMount(
            name="dshm",
            mount_path="/dev/shm")
        resources = client.V1ResourceRequirements(
            limits={
                'nvidia.com/gpu': 1
                # can add other resources like cpu, memory
            })
        capabilities = client.V1Capabilities(
            add=['SYS_PTRACE']
        )
        security_context = client.V1SecurityContext(
            capabilities=capabilities
        )
        tis_ports = [
            client.V1ContainerPort(container_port=ports[0], name="http-triton"),
            client.V1ContainerPort(container_port=ports[1], name="grpc-triton"),
            client.V1ContainerPort(container_port=ports[2], name="metrics-triton")
        ]
        container = client.V1Container(
            name="container",
            image=image,
            command=["/bin/bash", "-c"],
            args=[command],
            resources=resources,
            volume_mounts=[dshm_volume_mount],
            ports=tis_ports,
            security_context=security_context)

        dshm_volume = client.V1Volume(
            name="dshm",
            empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "purpose": "tao-toolkit-job",
                    "app": deployment_name,  # use deployment_name as the selector name
                }
            ),
            spec=client.V1PodSpec(
                image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
                containers=[container],
                volumes=[dshm_volume],
            ))
        spec = client.V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector={"matchLabels": {"app": deployment_name}})
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=spec)
        self.logger.info("Prepared deployment configs")
        try:
            api_instance.create_namespaced_deployment(
                body=deployment,
                namespace=name_space)
            self.logger.info("Start create deployment")
            return
        except Exception as e:
            self.logger.error(f"Create deployment got error: {e}")
            return

    def create_tis_service(self, tis_service_name, deploy_label, ports=(8000, 8001, 8002)):
        """Create TIS service"""
        name_space = self.get_namespace()
        tis_ports = [
            client.V1ServicePort(name="http", protocol="TCP", port=ports[0], target_port=ports[0]),
            client.V1ServicePort(name="grpc", protocol="TCP", port=ports[1], target_port=ports[1]),
            client.V1ServicePort(name="metrics", protocol="TCP", port=ports[2], target_port=ports[2]),
        ]
        spec = client.V1ServiceSpec(ports=tis_ports, selector={"app": deploy_label}, type="LoadBalancer")
        # add annotation, it will only works in Azure, but will not affect other cloud
        annotation = {
            "service.beta.kubernetes.io/azure-load-balancer-internal": "true"
        }
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=tis_service_name,
                labels={"app": tis_service_name},
                annotations=annotation
            ),
            spec=spec,
        )
        api_instance = client.CoreV1Api()
        self.logger.info("Prepared TIS Service configs")
        try:
            api_instance.create_namespaced_service(
                body=service,
                namespace=name_space)
            self.logger.info("Start create TIS Service")
            return
        except Exception as e:
            self.logger.error(f"Create TIS Service got error: {e}")
            return

    def get_triton_deployment_pods(self, deployment_name):
        """Returns pods of a Triton deployment"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()
        label_selector = f"app={deployment_name}"
        try:
            pods = api_instance.list_namespaced_pod(
                namespace=name_space,
                label_selector=label_selector)
            pods_ip = []
            for pod in pods.items:
                pods_ip.append(pod.status.pod_ip)
            return pods_ip
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return []

    def get_triton_deployment_status(self, deployment_name, replicas=1):
        """Returns status of Triton deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=name_space)
            available_replicas = api_response.status.available_replicas
            if not isinstance(available_replicas, int) or available_replicas < replicas:
                return {"status": "ReplicaNotReady"}
            return {"status": "Running"}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("Triton Deployment not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def get_tis_service_status(self, tis_service_name, ports=(8000, 8001, 8002)):
        """Returns status of TIS Service"""
        name_space = self.get_namespace()
        try:
            tis_service_ip = self.get_service_in_cluster_ip(tis_service_name, namespace=name_space)
            # need to double confirm if the service is ready
            url = f"http://{tis_service_ip}:{ports[0]}/v2/health/ready"
            try:
                endpoint_response = requests.get(url, timeout=120)
                if endpoint_response.status_code == 200:
                    return {"status": "Running", "tis_service_ip": tis_service_ip}
                return {"status": "NotReady"}
            except Exception as e:
                self.logger.error(f"Exception thrown in status_tis_service is {str(e)}")
                return {"status": "NotReady"}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("TIS Service not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def delete_triton_deployment(self, deployment_name):
        """Deletes a Triton deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.delete_namespaced_deployment(
                name=deployment_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            self.logger.info(f"Triton Deployment deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Triton Deployment failed to delete, got error: {e}")
            return

    def delete_tis_service(self, tis_service_name):
        """Deletes TIS service"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.delete_namespaced_service(
                name=tis_service_name,
                namespace=name_space,
            )
            self.logger.info(f"TIS Service deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"TIS Service failed to delete, got error: {e}")
            return

    def create_tensorboard_deployment(self, deployment_name, image, command, logs_image, logs_command, replicas):
        """Creates Tensorboard Deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        logs_volume_mount = client.V1VolumeMount(
            name="tb-data",
            mount_path="/tfevents")
        capabilities = client.V1Capabilities(
            add=['SYS_PTRACE']
        )
        security_context = client.V1SecurityContext(
            capabilities=capabilities
        )

        tb_port = [
            client.V1ContainerPort(container_port=6006)
        ]
        resources = client.V1ResourceRequirements(
            limits={
                'memory': "600Mi",
                'cpu': "10m",
            },
            requests={
                'memory': '300Mi',
                'cpu': "5m"
            }
        )
        no_gpu = client.V1EnvVar(
            name="NVIDIA_VISIBLE_DEVICES",
            value="none")
        mongo_secret_env = client.V1EnvVar(
            name="MONGOSECRET",
            value=mongo_secret  # pylint: disable=E0606
        )
        mongo_operator_enabled_env = client.V1EnvVar(
            name="MONGO_OPERATOR_ENABLED",
            value=str(mongo_operator_enabled).lower()
        )
        mongo_namespace_env = client.V1EnvVar(
            name="NAMESPACE",
            value=mongo_namespace
        )
        backend_env = client.V1EnvVar(
            name="BACKEND",
            value=self.backend,
        )
        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        tb_container = client.V1Container(
            name="tb-container",
            image=image,
            env=[no_gpu],
            command=["/bin/sh", "-c"],
            args=[command],
            resources=resources,
            volume_mounts=[logs_volume_mount],
            ports=tb_port,
            security_context=security_context)

        tb_logs_container = client.V1Container(
            name="tb-logs-container",
            image=logs_image,
            env=[no_gpu,
                 mongo_secret_env,
                 mongo_operator_enabled_env,
                 mongo_namespace_env,
                 backend_env],
            command=["/bin/sh", "-c"],
            resources=resources,
            args=[logs_command],
            volume_mounts=[logs_volume_mount],
            security_context=security_context,
        )

        logs_volume = client.V1Volume(
            name="tb-data",
            empty_dir=client.V1EmptyDirVolumeSource())

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "purpose": "tao-toolkit-tensorboard-job",
                    "resource-type": "tensorboard",
                    "app": deployment_name,  # use deployment_name as the selector name
                }
            ),
            spec=client.V1PodSpec(
                containers=[tb_container, tb_logs_container],
                volumes=[logs_volume],
                image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)]
            ))

        spec = client.V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector={"matchLabels": {"app": deployment_name}})

        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name, labels={
                "resource-type": "tensorboard",
            }, owner_references=[self.get_owner_reference()]),
            spec=spec)

        self.logger.info("Prepared deployment configs")
        try:
            api_instance.create_namespaced_deployment(
                body=deployment,
                namespace=name_space)
            self.logger.info("Start create deployment")
            return
        except Exception as e:
            self.logger.error(f"Create deployment got error: {e}")
            return

    def create_tensorboard_service(self, tb_service_name, deploy_label):
        """Creates Tensorboard Service"""
        name_space = self.get_namespace()
        tb_port = [
            client.V1ServicePort(name='tb-default-port', port=6006, target_port=6006, protocol="TCP")
        ]
        spec = client.V1ServiceSpec(ports=tb_port, selector={"app": deploy_label})
        # add annotation, it will only works in Azure, but will not affect other cloud
        annotation = {
            "service.beta.kubernetes.io/azure-load-balancer-internal": "true",
        }
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=tb_service_name,
                labels={
                    "app": tb_service_name,
                    "resource-type": "tensorboard",
                },
                owner_references=[self.get_owner_reference()],
                annotations=annotation),
            spec=spec,
        )
        api_instance = client.CoreV1Api()

        self.logger.info("Prepared Tensorboard Service configs")
        try:
            api_instance.create_namespaced_service(
                body=service,
                namespace=name_space)
            self.logger.info("Start create Tensorboard Service")
            return
        except Exception as e:
            self.logger.error(f"Create Tensorboard Service got error: {e}")
            return

    def create_tensorboard_ingress(self, tb_service_name, tb_ingress_name, tb_ingress_path):
        """Creates Tensorboard Ingress"""
        name_space = self.get_namespace()
        networking_v1_api = client.NetworkingV1Api()
        ingress = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(
                name=tb_ingress_name,
                namespace=name_space,
                labels={
                    "resource-type": "tensorboard",
                },
                annotations={
                    "kubernetes.io/ingress.class": "nginx",
                    "nginx.ingress.kubernetes.io/client-max-body-size": "0m",
                    "nginx.ingress.kubernetes.io/proxy-body-size": "0m",
                    "nginx.ingress.kubernetes.io/body-size": "0m",
                    "nginx.ingress.kubernetes.io/client-body-buffer-size": "50m",
                    "nginx.ingress.kubernetes.io/proxy-buffer-size": "128k",
                    "nginx.ingress.kubernetes.io/proxy-buffers-number": "4",
                    "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
                    "meta.helm.sh/release-name": self.release_name,
                    "meta.helm.sh/release-namespace": name_space
                },
                owner_references=[self.get_owner_reference()]),
            spec=client.V1IngressSpec(
                rules=[client.V1IngressRule(
                    http=client.V1HTTPIngressRuleValue(
                        paths=[client.V1HTTPIngressPath(
                            path=tb_ingress_path,
                            path_type="Prefix",
                            backend=client.V1IngressBackend(
                                service=client.V1IngressServiceBackend(
                                    port=client.V1ServiceBackendPort(
                                        name='tb-default-port'
                                    ),
                                    name=tb_service_name
                                )
                            )
                        )]
                    )
                )]
            )
        )

        try:
            networking_v1_api.create_namespaced_ingress(
                body=ingress,
                namespace=name_space
            )
            self.logger.info("Created Tensorboard Ingress")
            return
        except Exception as e:
            self.logger.error(f"Create Tensorboard Ingress got error: {e}")
            return

    def get_tensorboard_deployment_status(self, deployment_name, replicas=1):
        """Returns status of Tensorboard deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=name_space)
            available_replicas = api_response.status.available_replicas
            if not isinstance(available_replicas, int) or available_replicas < replicas:
                return {"status": "ReplicaNotReady"}
            return {"status": "Running"}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("Tensorboard Deployment not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def get_tensorboard_service_status(self, tb_service_name, port=6006):
        """Returns status of TB Service"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()

        try:
            api_response = api_instance.read_namespaced_service(
                name=tb_service_name,
                namespace=name_space,
            )
            self.logger.info(f'TB Service API Response: {api_response}')
            tb_service_ip = api_response.spec.cluster_ip
            return {"status": "Running", "tb_service_ip": tb_service_ip}
        except ApiException as e:
            if e.status == 404:
                self.logger.info("TIS Service not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def delete_tensorboard_deployment(self, deployment_name):
        """Deletes Tensorboard Deployment"""
        name_space = self.get_namespace()
        api_instance = client.AppsV1Api()
        try:
            api_response = api_instance.delete_namespaced_deployment(
                name=deployment_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            self.logger.info(f"Tensorboard Deployment deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Deployment failed to delete, got error: {e}")
            return

    def delete_tensorboard_service(self, tb_service_name):
        """Deletes Tensorboard service"""
        name_space = self.get_namespace()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.delete_namespaced_service(
                name=tb_service_name,
                namespace=name_space,
            )
            self.logger.info(f"Tensorboard Service deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Service failed to delete, got error: {e}")
            return

    def delete_tensorboard_ingress(self, tb_ingress_name):
        """Delete Tensorboard Ingress"""
        name_space = self.get_namespace()
        networking_v1_api = client.NetworkingV1Api()
        try:
            api_response = networking_v1_api.delete_namespaced_ingress(
                name=tb_ingress_name,
                namespace=name_space
            )
            self.logger.info(f"Tensorboard Ingress deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Tensorboard Ingress failed to delete, got error: {e}")
            return
