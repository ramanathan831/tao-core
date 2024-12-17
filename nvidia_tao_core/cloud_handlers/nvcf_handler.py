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

"""Handler functions to manage NVCF related operations"""

import sys
import time
import requests
import functools

NUM_OF_RETRY = 3


class ErrorResponse:
    """Custom error response object"""

    def __init__(self, status_code, message="An error occurred"):
        """Initialize the ErrorResponse object.

        Args:
            status_code (int): The HTTP status code to represent the error.
            message (str, optional): A message providing additional context about the error.
                                     Defaults to "An error occurred".
        """
        self.status_code = status_code
        self.ok = False
        self.message = message

    def json(self):
        """Return a JSON-like dictionary for the error response.

        Returns:
            dict: A dictionary containing the status code and error message.
        """
        return {
            "status_code": self.status_code,
            "message": self.message
        }

    def __repr__(self):
        """Return a string representation of the ErrorResponse object.

        Returns:
            str: A string that shows the status code and error message.
        """
        return f"<ErrorResponse(status_code={self.status_code}, message={self.message})>"


def retry_method(response=False):
    """Retry Cloud storage methods for NUM_RETRY times"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(NUM_OF_RETRY):
                try:
                    # Call the actual function
                    result = func(*args, **kwargs)
                    # If response handling is enabled, check for `ok` status
                    if response:
                        if result.ok:
                            return result
                        print(f"Response not OK (attempt {attempt + 1}): {result.status_code}", file=sys.stderr)
                        time.sleep(30)  # Wait between retries
                    else:
                        # If no response-based retry, just return the result
                        return result
                except Exception as e:
                    # Log or handle the exception
                    print(f"Exception in {func.__name__} on attempt {attempt + 1}: {e}", file=sys.stderr)
                    time.sleep(30)  # Wait between retries

            # After retries, return error response or raise an error based on decorator parameter
            if response:
                return ErrorResponse(status_code=404, message=f"Failed to execute {func.__name__} after {NUM_OF_RETRY} retries")
            raise ValueError(f"Failed to execute {func.__name__} after {NUM_OF_RETRY} retries")
        return wrapper
    return decorator


@retry_method(response=True)
def invoke_function(deployment_string, api_endpoint="", kind="", handler_id="", is_job=True, job_id="", request_body={}, ngc_key=""):
    """Invoke a NVCF function"""
    request_metadata = {"api_endpoint": api_endpoint,
                        "kind": kind,
                        "handler_id": handler_id,
                        "is_job": is_job,
                        "job_id": job_id,
                        "request_body": request_body,
                        "ngc_key": ngc_key,
                        }
    if api_endpoint == "status_update":
        request_metadata["is_json_request"] = True

    function_id, version_id = deployment_string.split(":")

    url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{function_id}/versions/{version_id}"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {ngc_key}",
    }

    try:
        response = requests.post(url, headers=headers, json=request_metadata, timeout=120)
    except Exception as e:
        print("Exception caught during invoking NVCF function", deployment_string, e, file=sys.stderr)
        raise e

    if not response.ok:
        print("Invocation failed.", file=sys.stderr)
        print("Response status code:", response.status_code, file=sys.stderr)
        print("Response content:", response.text, file=sys.stderr)
    return response
