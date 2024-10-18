# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Static tests execution"""

import argparse
import subprocess
import sys

RCFILE = ".pylintrc"
TEST_MODULES = [
    "nvidia_tao_core",
]
STATIC_TESTS = [
    f"pylint --rcfile {RCFILE}",
    "pydocstyle --ignore=D400,D213,D203,D211,D4",
    "flake8 --ignore=E24,W504,E501",
]


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(prog="run_tests_local", description="Simple script to run tests.")
    parser.add_argument("--tag", default=None, help="Tag to the local base docker image.", type=str)
    return vars(parser.parse_args(args))


def main(cl_args=None):
    """Simple function to run local tests."""
    test_command = []
    for test in STATIC_TESTS:
        test_command.extend(["{} {}".format(test, " ".join(TEST_MODULES))])

    for command in test_command:
        sys.stdout.flush()
        rc = subprocess.call(command, stdout=sys.stdout, shell=True)
        assert rc == 0


if __name__ == "__main__":
    main()
