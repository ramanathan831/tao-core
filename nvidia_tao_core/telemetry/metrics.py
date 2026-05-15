#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# CONFIDENTIAL! DO NOT SPREAD!
# DO NOT CHANGE THIS FILE!

"""Authentication utils metrics modules"""
import json
import requests

__c29tZSByYW5kb20gc3RyaW5n = b'TEVUIE1FIElO'


def report(data={}, base_url='https://api.tao.ngc.nvidia.com', timeout=10):
    """report metrics"""
    url = f'{base_url}/api/v1/metrics'
    if isinstance(data, dict):
        data = json.dumps(data)
    resp = requests.post(url, data=data, auth=('$metricstoken', __c29tZSByYW5kb20gc3RyaW5n), timeout=timeout)
    if resp.status_code == 201:
        return None
    return f'error {resp.status_code}'
