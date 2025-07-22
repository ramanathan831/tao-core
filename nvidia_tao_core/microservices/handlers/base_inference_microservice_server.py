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

"""Base Inference Microservice Server - Abstract base class for persistent model servers in StatefulSet containers

Provides common functionality for loading models, serving inference requests, and managing server lifecycle
"""

import json
import os
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Tuple
from flask import Flask, request, jsonify

from nvidia_tao_core.microservices.handlers.container_handler import prepare_data_before_job_run
from nvidia_tao_core.cloud_handlers.utils import download_from_user_storage, get_file_path_from_cloud_string
from nvidia_tao_core.microservices.utils import safe_load_file


class BaseInferenceMicroserviceServer(ABC):
    """Abstract base class for TAO model servers in StatefulSet containers"""

    def __init__(self, job_id: str, port: int = 8080, cloud_storage=None, **model_params):
        """Initialize base model server

        Args:
            job_id: Unique job identifier
            port: Server port (default 8080)
            cloud_storage: Cloud storage configuration
            **model_params: Model-specific parameters (e.g., model_path, etc.)
        """
        self.setup_logging()
        self.job_id = job_id
        self.port = port
        self.model = None
        self.model_loaded = False
        self.cloud_storage = cloud_storage
        self.model_state_dir = "/tmp/tao_models"
        self.model_params = model_params  # Store all model-specific params

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f'tao_{self.__class__.__name__.lower()}')

    def save_model_state(self, loaded: bool = False, load_time: float = None, error: str = None):
        """Save model loading state to file

        Args:
            loaded: Whether model is loaded successfully
            load_time: Time taken to load model
            error: Error message if loading failed
        """
        model_state = {
            "job_id": self.job_id,
            "model_params": self.model_params,
            "loaded": loaded,
            "timestamp": datetime.now().isoformat(),
            "server_port": self.port,
            "model_type": self.__class__.__name__
        }

        if load_time is not None:
            model_state["load_time"] = load_time
        if error:
            model_state["error"] = error

        os.makedirs(self.model_state_dir, exist_ok=True)
        state_file = f"{self.model_state_dir}/{self.job_id}_server.json"

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(model_state, f, indent=2)

    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state from file

        Returns:
            Model state dictionary or empty dict if file doesn't exist
        """
        state_file = f"{self.model_state_dir}/{self.job_id}_server.json"
        if os.path.exists(state_file):
            return safe_load_file(state_file)
        return {}

    @abstractmethod
    def load_model_into_memory(self, **kwargs) -> bool:
        """Load the specific model implementation

        Args:
            **kwargs: Model-specific configuration parameters

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def run_model_inference(self, **kwargs) -> Dict[str, Any]:
        """Run model-specific inference

        Args:
            **kwargs: All inference parameters (model-specific naming)

        Returns:
            Inference results dictionary
        """
        pass

    @abstractmethod
    def get_supported_file_extensions(self) -> Tuple[List[str], List[str]]:
        """Get supported file extensions for this model

        Returns:
            Tuple of (image_extensions, video_extensions)
        """
        pass

    def load_model(self, **kwargs) -> bool:
        """Load model with error handling and state management

        Args:
            **kwargs: Model-specific configuration parameters

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading {self.__class__.__name__} model")
            start_time = time.time()

            # Save initial state
            self.save_model_state(loaded=False)

            # Merge model_params with provided kwargs
            all_params = {**self.model_params, **kwargs}

            # Call model-specific loading implementation
            success = self.load_model_into_memory(**all_params)

            if success:
                load_time = time.time() - start_time
                self.model_loaded = True
                print(f"{self.__class__.__name__} model loaded successfully in {load_time:.2f} seconds")
                self.save_model_state(loaded=True, load_time=load_time)
            else:
                self.save_model_state(loaded=False, error="Model loading failed")

            return success

        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            self.logger.error(error_msg)
            self.model_loaded = False
            self.save_model_state(loaded=False, error=str(e))
            return False

    def download_and_process_file(self, input_file: str) -> str:
        """Download file from cloud storage if needed and return local path

        Args:
            input_file: Input file path or cloud URL

        Returns:
            Local file path
        """
        try:
            # Handle cloud storage URLs by downloading first
            _, _, cloud_file_path = get_file_path_from_cloud_string(input_file)
            actual_file_path = download_from_user_storage(
                cloud_storage=self.cloud_storage,
                job_id=self.job_id,
                value=cloud_file_path if self.cloud_storage else input_file,
                dictionary={},
                key="",
                preserve_source_path=True,
                reset_value=False
            )
            return actual_file_path

        except Exception as e:
            self.logger.error(f"Failed to process {input_file}: {e}")
            raise

    def run_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference with error handling and timing

        Args:
            **kwargs: All inference parameters (model-specific)

        Returns:
            Inference results dictionary
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            start_time = time.time()

            # Call model-specific inference implementation
            result = self.run_model_inference(**kwargs)

            inference_time = time.time() - start_time

            # Add common metadata to result
            if isinstance(result, dict):
                result.update({
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": self.__class__.__name__
                })

            self.logger.info(f"Inference completed in {inference_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

    def create_flask_app(self):
        """Create Flask app with common endpoints"""
        app = Flask(__name__)

        @app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "model_loaded": self.model_loaded,
                "job_id": self.job_id,
                "model_type": self.__class__.__name__
            })

        @app.route('/status', methods=['GET'])
        def status():
            """Detailed status endpoint"""
            model_state = self.get_model_state()
            return jsonify({
                "job_id": self.job_id,
                "model_loaded": self.model_loaded,
                "model_state": model_state,
                "server_port": self.port
            })

        @app.route('/inference', methods=['POST'])
        def inference():
            """Inference endpoint"""
            try:
                # Check if model is loaded
                model_state = self.get_model_state()
                if not model_state.get("loaded", False):
                    response_data = {
                        "job_id": self.job_id,
                        "status": "waiting",
                        "message": "Model not loaded yet, try again later"
                    }
                    return jsonify(response_data)

                # Parse request data - pass all parameters to model implementation
                data = request.json
                if not data:
                    return jsonify({"error": "Request body is required"}), 400

                # Run inference with all parameters
                results = self.run_inference(**data)

                response_data = {
                    "status": "completed",
                    "results": results,
                    "message": f"{self.__class__.__name__} inference completed"
                }

                return jsonify(response_data)

            except Exception as e:
                self.logger.error(f"Inference request failed: {e}")
                return jsonify({
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        return app

    def start_server(self):
        """Start the persistent model server"""
        if not self.model_loaded:
            self.logger.error("Cannot start server: model not loaded")
            return False

        try:
            app = self.create_flask_app()
            self.logger.info(f"Starting {self.__class__.__name__} Server on port {self.port}")
            app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    @classmethod
    def create_from_tao_job(cls, job_data: Dict[str, Any], docker_env_vars: Dict[str, Any],
                            port: int = 8080):
        """Factory method to create model server from TAO job data

        Args:
            job_data: TAO job metadata
            docker_env_vars: Docker environment variables
            port: Server port

        Returns:
            Tuple of (configured model server instance, specs dict)
        """
        cloud_storage, specs, _ = prepare_data_before_job_run(job_data, docker_env_vars)

        return cls(
            job_id=job_data["job_id"],
            port=port,
            cloud_storage=cloud_storage,
            **specs  # Pass all specs as model-specific kwargs
        )
