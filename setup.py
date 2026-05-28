# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup script to build TAO-Core."""

import setuptools

from release.python.utils import utils

PACKAGE_LIST = [
    "nvidia_tao_core"
]

def read_requirements():
    """Read dependencies from requirements-pip.txt."""
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


version_locals = utils.get_version_details()
setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.py', "*.pyc", "*.yaml", "*.so", "*.pdf"],
        'nvidia_tao_core.microservices': [
            'pretrained_models.csv',
            'utils/specs_utils/specs/**/*.csv',
            '*.sh',
            'uwsgi.ini',
            'handlers/network_configs/*',
            'nginx.conf',
            'blueprints/v1/templates/**/*.html',
            'blueprints/v2/templates/**/*.html',
            'api_spec/*.json',
        ]
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'finetuning-microservice=nvidia_tao_core.microservices.app:main',
            'get-microservice-script=nvidia_tao_core.microservices.utils:print_start_script_path',
            'get-nginx-conf-path=nvidia_tao_core.microservices.utils:print_nginx_conf_path',
        ],
    },
)

