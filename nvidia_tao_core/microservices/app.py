#!/usr/bin/env python3

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

"""TAO Core Flask application with API versioning support."""

import os
import sys
import json
import logging
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.profiler import ProfilerMiddleware
from flask import Flask, jsonify, make_response
from flask_wtf.csrf import CSRFProtect
from marshmallow import exceptions

from .config import FLASK_CONFIG
from .api_versions import (
    register_all_api_versions,
    initialize_all_apispec_version,
    register_all_routes_with_apispec,
    register_all_schemas_with_apispec
)
from .blueprints.v1.schemas import ErrorRsp as ErrorRspSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomProfilerMiddleware(ProfilerMiddleware):
    """Custom profiler middleware to exclude health check endpoints."""

    def __call__(self, environ, start_response):
        """Handle WSGI request, skipping profiling for health check endpoints."""
        if '/api/v1/health' in environ['PATH_INFO']:
            return self._app(environ, start_response)
        if '/api/v2/health' in environ['PATH_INFO']:
            return self._app(environ, start_response)
        return super().__call__(environ, start_response)


def create_app():
    """Create and configure the versioned Flask application."""
    app = Flask(__name__)

    # Apply configuration
    app.config.update(FLASK_CONFIG)
    app.json.sort_keys = False

    # Initialize CSRF protection
    csrf = CSRFProtect()
    csrf.init_app(app)

    # Configure profiler if enabled
    if os.getenv("PROFILER", "FALSE") == "True":
        app.config["PROFILE"] = True
        app.wsgi_app = CustomProfilerMiddleware(
            app.wsgi_app,
            stream=sys.stderr,
            sort_by=('cumtime',),
            restrictions=[50],
        )

    # Register error handlers
    @app.errorhandler(HTTPException)
    def handle_exception(e):
        """Return JSON instead of HTML for HTTP errors."""
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    @app.errorhandler(exceptions.ValidationError)
    def handle_validation_exception(e):
        """Return 400 bad request for validation exceptions"""
        metadata = {"error_desc": str(e)}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    # Register root-level endpoints
    @app.route('/')
    def root():
        """Root endpoint."""
        return jsonify({
            "message": "TAO Core API",
            "version": "1.0.0",
            "available_versions": ["v1", "v2"]
        })

    @app.route('/version')
    def version():
        """Version endpoint."""
        return jsonify({
            "version": "1.0.0",
            "api_versions": ["v1", "v2"]
        })

    @app.route('/openapi.json')
    def openapi_json():
        """OpenAPI JSON endpoint."""
        return jsonify({
            "openapi": "3.0.0",
            "info": {
                "title": "TAO Core API",
                "version": "1.0.0"
            },
            "paths": {}
        })

    # Register all versioned API blueprints
    logger.info("Register all versioned API blueprints")
    register_all_api_versions(app)

    # Initialize all versioned OpenAPI specs
    logger.info("Initialize all versioned OpenAPI Specs")
    initialize_all_apispec_version(app)

    # Register schemas with appropriate OpenAPI specs
    logger.info("Register schemas with versioned OpenAPI specs")
    register_all_schemas_with_apispec(app)

    # Register routes with appropriate OpenAPI specs
    logger.info("Register routes with versioned OpenAPI specs")
    register_all_routes_with_apispec(app)

    logger.info("TAO Core Flask application created and configured successfully")
    return app


# Create the application instance


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
