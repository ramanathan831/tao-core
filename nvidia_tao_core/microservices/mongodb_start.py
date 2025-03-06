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

"""MongoDB init script"""
import time
from kubernetes import client, config
import sys
import os
from time import sleep
from pymongo import MongoClient
from urllib import parse

mongodb_crd_group = 'mongodbcommunity.mongodb.com'
mongodb_crd_version = 'v1'
mongodb_crd_plural = 'mongodbcommunity'
mongodb_crd_name = 'mongodb'
mongodb_desired_replica_count = int(os.getenv('MONGODESIREDREPLICAS', '3'))
mongodb_storage_class = os.getenv('MONGOSTORAGECLASS', 'nfs-client')
mongodb_storage_size = os.getenv('MONGOSTORAGESIZE', '100Gi')
mongodb_storage_access_mode = os.getenv('MONGOSTORAGEACCESSMODE', 'ReadWriteOnce')
mongod_memory_request = os.getenv('MONGODMEMORYREQUEST', '5000M')
mongod_cpu_request = os.getenv('MONGODCPUREQUEST', '1000m')
mongod_cpu_limit = os.getenv('MONGODCPULIMIT', '4000m')
mongod_memory_limit = os.getenv('MONGODMEMORYLIMIT', '8000M')
mongo_agent_memory_request = os.getenv('MONGOAGENTMEMORYREQUEST', '400M')
mongo_agent_cpu_request = os.getenv('MONGOAGENTCPUREQUEST', '500m')
mongo_agent_cpu_limit = os.getenv('MONGOAGENTCPULIMIT', '1000m')
mongo_agent_memory_limit = os.getenv('MONGOAGENTMEMORYLIMIT', '500M')
mongodb_namespace = os.getenv("NAMESPACE", "default")
mongo_operator_enabled = os.getenv('MONGO_OPERATOR_ENABLED', 'true') == 'true'
mongo_secret = os.getenv("MONGOSECRET")
encoded_secret = parse.quote(mongo_secret, safe='')


def create_mongodb_replicaset():
    """Creates a MongoDB replicaset"""
    config.load_incluster_config()

    # Initialize the custom objects API
    api_instance = client.CustomObjectsApi()

    mongodb_body = {
        "apiVersion": "mongodbcommunity.mongodb.com/v1",
        "kind": "MongoDBCommunity",
        "metadata": {
            "name": mongodb_crd_name,
        },
        "spec": {
            "members": mongodb_desired_replica_count,
            "type": "ReplicaSet",
            "version": "6.0.5",
            "security": {
                "authentication": {
                    "modes": ["SCRAM"]
                }
            },
            "users": [
                {
                    "name": "default-user",
                    "db": "admin",
                    "passwordSecretRef": {
                        "name": "mongodb-keyfile-secret",
                        "key": "mongodb-keyfile"
                    },
                    "roles": [
                        {
                            "name": "root",
                            "db": "admin"
                        }
                    ],
                    "scramCredentialsSecretName": "scram-secret",
                }
            ],
            "additionalMongodConfig": {
                "storage.wiredTiger.engineConfig.journalCompressor": "zlib"
            },
            "statefulSet": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "mongodb-agent",
                                    "resources": {
                                        "requests": {
                                            "cpu": mongo_agent_cpu_request,
                                            "memory": mongo_agent_memory_request
                                        },
                                        "limits": {
                                            "cpu": mongo_agent_cpu_limit,
                                            "memory": mongo_agent_memory_limit
                                        }
                                    }
                                },
                                {
                                    "name": "mongod",
                                    "resources": {
                                        "requests": {
                                            "cpu": mongod_cpu_request,
                                            "memory": mongod_memory_request
                                        },
                                        "limits": {
                                            "cpu": mongod_cpu_limit,
                                            "memory": mongod_memory_limit
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "volumeClaimTemplates": [
                        {
                            "metadata": {
                                "name": "data-volume"
                            },
                            "spec": {
                                "storageClassName": mongodb_storage_class,
                                "accessModes": [mongodb_storage_access_mode],
                                "resources": {
                                    "requests": {
                                        "storage": mongodb_storage_size
                                    }
                                }
                            }
                        },
                        {
                            "metadata": {
                                "name": "logs-volume"
                            },
                            "spec": {
                                "storageClassName": mongodb_storage_class,
                                "accessModes": [mongodb_storage_access_mode],
                                "resources": {
                                    "requests": {
                                        "storage": mongodb_storage_size
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
    }

    # Create the custom resource in the specified namespace
    try:
        api_instance.create_namespaced_custom_object(
            mongodb_crd_group,
            mongodb_crd_version,
            mongodb_namespace,
            mongodb_crd_plural,
            mongodb_body
        )
        print("MongoDB replicaset created successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Failed to create MongoDB replicaset': {e}", file=sys.stderr)
        raise e


def patch_mongodb_replicaset():
    """Patch MongoDB Replicaset"""
    config.load_incluster_config()

    # Initialize the custom objects API
    api_instance = client.CustomObjectsApi()

    updated_spec = {
        "spec": {
            "members": mongodb_desired_replica_count
        }
    }

    try:
        api_instance.patch_namespaced_custom_object(
            mongodb_crd_group,
            mongodb_crd_version,
            mongodb_namespace,
            mongodb_crd_plural,
            name=mongodb_crd_name,
            body=updated_spec
        )
        print("MongoDB replicaset patched successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Failed to patch MongoDB replicaset': {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    print(f"Mongo Operator Enabled: {mongo_operator_enabled}", file=sys.stderr)
    if not mongo_operator_enabled:
        try:
            from handlers.mongo_handler import MongoHandler
            # try to make a test connection to DB replicaset
            mongo_experiments = MongoHandler(
                "tao",
                "experiments",
                retries=1
            )
        except Exception as e:  # if error, initialize replicaset
            print(f"Exception caught in mongodb start with message {str(e)}", file=sys.stderr)
            retry = 0
            while retry <= 10:
                try:
                    print("Retrying attempt: ", retry, file=sys.stderr)
                    init_time = 20 * mongodb_desired_replica_count
                    time.sleep(init_time)
                    connection_string = (
                        f'mongodb://default-user:{encoded_secret}@'
                        f'mongodb-0.mongodb-svc.{mongodb_namespace}.svc.cluster.local'
                        '?authSource=admin'
                    )
                    c = MongoClient(connection_string, directConnection=True)
                    rs_members = []
                    for rs_id in range(mongodb_desired_replica_count):
                        rs_members.append({
                            '_id': rs_id,
                            'host': f'mongodb-{rs_id}.mongodb-svc.{mongodb_namespace}.svc.cluster.local:27017'
                        })
                    mongo_config = {
                        '_id': 'mongodb',
                        'members': rs_members
                    }
                    print("Going to run replSetInitiate command", file=sys.stderr)
                    out = c.admin.command("replSetInitiate", mongo_config)
                    print("replSetInitiate output", out, file=sys.stderr)
                    print("Breaking while", file=sys.stderr)
                    print(out, file=sys.stderr)
                    break
                except Exception as e:
                    print(f"Error initializing replicaset! {e}", file=sys.stderr)
                    retry += 1
                    print(f"Retrying {retry}", file=sys.stderr)
                    if retry > 10:
                        import traceback
                        print("traceback", traceback.format_exc(), file=sys.stderr)
                        print("Raise exception", file=sys.stderr)
                        raise e
    else:
        config.load_incluster_config()

        api_instance = client.CustomObjectsApi()

        try:
            api_response = api_instance.list_namespaced_custom_object(
                group=mongodb_crd_group,
                version=mongodb_crd_version,
                namespace=mongodb_namespace,
                plural=mongodb_crd_plural
            )
            print("MongoDB response: ", api_response, file=sys.stderr)
            items = api_response['items']
            if len(items) == 0:
                create_mongodb_replicaset()
            else:
                status = items[0].get('status', {})
                phase = status.get('phase', '')
                if phase == 'Running' and mongodb_desired_replica_count != status.get('currentStatefulSetReplicas', 0):
                    patch_mongodb_replicaset()
                elif phase != 'Running':
                    print("Unknown MongoDB Replicaset status: ", status, file=sys.stderr)
                    raise Exception("Unknown MongoDB Replicaset status")

            # Wait for all replicas to be ready
            replica_count = 0
            phase = 'Pending'
            while replica_count != mongodb_desired_replica_count and phase != 'Running':
                api_response = api_instance.list_namespaced_custom_object(
                    group=mongodb_crd_group,
                    version=mongodb_crd_version,
                    namespace=mongodb_namespace,
                    plural=mongodb_crd_plural
                )
                print("MongoDB response: ", api_response, file=sys.stderr)
                items = api_response.get('items', [])
                if items:
                    status = items[0].get('status', {})
                    phase = status.get('phase', 'Pending')
                    replica_count = status.get('currentStatefulSetReplicas', 0)
                    print(
                        f"Current Replica Count: {replica_count}, waiting for "
                        f"{mongodb_desired_replica_count - replica_count} more replicas to be ready",
                        file=sys.stderr
                    )
                    print(f"Current ReplicaSet Phase {phase}", file=sys.stderr)
                sleep(20)

        except Exception as e:
            print("Exception when calling CustomObjectsApi: ", str(e), file=sys.stderr)
            raise e
