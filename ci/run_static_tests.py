# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    "flake8 --ignore=E24,W504 --max-line-length=120",
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
