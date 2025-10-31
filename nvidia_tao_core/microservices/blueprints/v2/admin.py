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

"""Admin blueprint for v2 API - documentation, metrics, and utility endpoints."""

import os
import re
import bson
import shutil
import logging
from datetime import datetime
from flask import Blueprint, jsonify, make_response, render_template, send_file
from flask import current_app, request

from nvidia_tao_core.microservices.decorators import disk_space_check
from .schemas import ErrorRsp, TelemetryReq
from nvidia_tao_core.microservices.config import get_tao_version
from nvidia_tao_core.microservices.utils.stateless_handler_utils import set_metrics, get_metrics, get_root
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file

TIMEOUT = 240
logger = logging.getLogger(__name__)

# v2 Admin Blueprint - URL prefix will be set during registration (/api/v2)
admin_bp_v2 = Blueprint('admin_v2', __name__, template_folder='templates')

tao_version = get_tao_version()


@admin_bp_v2.route('/', methods=['GET'])
def version_v2():
    """api v2 root endpoint"""
    return make_response(jsonify([
        'login',
        'auth',
        'health',
        'openapi.yaml',
        'openapi.json',
        'rapipdf',
        'redoc',
        'swagger',
        'version',
        'tao_api_notebooks.zip',
        'orgs',
        'metrics'
    ]))


@admin_bp_v2.route('/orgs', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing orgs is not authorized: Missing Org Name", "error_code": 1}
    schema = ErrorRsp()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@admin_bp_v2.route('/orgs/<org_name>', methods=['GET'])
@disk_space_check
def user(org_name):
    """user endpoint"""
    return make_response(jsonify(['dataset', 'experiment']))


@admin_bp_v2.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    spec = current_app.spec_v2
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@admin_bp_v2.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    spec = current_app.spec_v2
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@admin_bp_v2.route('/rapipdf', methods=['GET'])
def rapipdf():
    """rapipdf endpoint"""
    return render_template('v2/rapipdf.html')


@admin_bp_v2.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('v2/redoc.html')


@admin_bp_v2.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    logger.info(
        f"Received swagger request for v2 API, root_path={admin_bp_v2.root_path}, "
        f"template_folder={admin_bp_v2.template_folder}")
    return render_template('v2/swagger.html')


@admin_bp_v2.route('/version', methods=['GET'])
def version():
    """version endpoint"""
    git_branch = os.environ.get('GIT_BRANCH', 'unknown')
    git_commit_sha = os.environ.get('GIT_COMMIT_SHA', 'unknown')
    git_commit_time = os.environ.get('GIT_COMMIT_TIME', 'unknown')
    version = {'version': tao_version, 'branch': git_branch, 'sha': git_commit_sha, 'time': git_commit_time}
    r = make_response(jsonify(version))
    r.mimetype = 'application/json'
    return r


@admin_bp_v2.route('/tao_api_notebooks.zip', methods=['GET'])
@disk_space_check
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "/shared/notebooks/")

    # Return the zip file as a downloadable attachment
    return send_file("/tmp/tao_api_notebooks.zip", as_attachment=True,
                     download_name="tao_api_notebooks.zip", mimetype='application/zip')


@admin_bp_v2.route('/metrics', methods=['POST'])
def metrics_upsert():
    """Report execution of new action."""
    now = old_now = datetime.now()

    # get action report

    try:
        data = TelemetryReq().load(request.get_json(force=True))
    except Exception as e:
        logger.error("Exception thrown in metrics_upsert: %s", str(e))
        return make_response(jsonify({}), 400)

    # update metrics.json

    metrics = get_metrics()
    if not metrics:
        metrics = safe_load_file(os.path.join(get_root(), 'metrics.json'))
        if not metrics:
            metadata = {
                "error_desc": "Metrics.json file not exists or can not be updated now, please try again later.",
                "error_code": 503
            }
            schema = ErrorRsp()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
            return response

    old_now = datetime.fromisoformat(metrics.get('last_updated', now.isoformat()))
    version = re.sub("[^a-zA-Z0-9]", "_", data.get('version', 'unknown')).lower()
    action = re.sub("[^a-zA-Z0-9]", "_", data.get('action', 'unknown')).lower()
    network = re.sub("[^a-zA-Z0-9]", "_", data.get('network', 'unknown')).lower()
    success = data.get('success', False)
    time_lapsed = data.get('time_lapsed', 0)
    gpus = data.get('gpu', ['unknown'])
    if success:
        metrics[f'total_action_{action}_pass'] = metrics.get(f'total_action_{action}_pass', 0) + 1
    else:
        metrics[f'total_action_{action}_fail'] = metrics.get(f'total_action_{action}_fail', 0) + 1
    metrics[f'version_{version}_action_{action}'] = metrics.get(f'version_{version}_action_{action}', 0) + 1
    metrics[f'network_{network}_action_{action}'] = metrics.get(f'network_{network}_action_{action}', 0) + 1
    metrics['time_lapsed_today'] = metrics.get('time_lapsed_today', 0) + time_lapsed
    if now.strftime("%d") != old_now.strftime("%d"):
        metrics['time_lapsed_today'] = time_lapsed
    for gpu in gpus:
        gpu = re.sub("[^a-zA-Z0-9]", "_", gpu).lower()
        metrics[f'gpu_{gpu}_action_{action}'] = metrics.get(f'gpu_{gpu}_action_{action}', 0) + 1
    metrics['last_updated'] = now.isoformat()

    def sanitize_gpu_name(gpu_name):
        # Convert to uppercase first, then replace all non-alphanumeric characters with _
        return re.sub("[^a-zA-Z0-9]", "_", gpu_name.upper())

    def create_gpu_identifier(gpu_list):
        # Count occurrences of each GPU type (case insensitive)
        gpu_counts = {}
        for gpu in map(sanitize_gpu_name, gpu_list):
            gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1

        # Format as "gpu_count_gpu1_count_gpu2_count..."
        gpu_parts = [f"{gpu}_{count}" for gpu, count in sorted(gpu_counts.items())]
        return f"{len(gpu_list)}_{'_'.join(gpu_parts)}"

    # Build metric name with all attributes
    status = "pass" if success else "fail"
    metric_components = [
        "network", network,
        "action", action,
        "version", version,
        "status", status,
        "gpu", create_gpu_identifier(gpus)
    ]
    full_metric_name = "_".join(metric_components)

    # Update metric counter
    metrics[full_metric_name] = metrics.get(full_metric_name, 0) + 1

    set_metrics(metrics)

    # success

    return make_response(bson.json_util.dumps(metrics), 201)
