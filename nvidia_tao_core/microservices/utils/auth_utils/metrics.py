#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication utils metrics modules"""
import json
import requests

__c29tZSByYW5kb20gc3RyaW5n = b'TEVUIE1FIElO'


def validate(encoded_key=b''):
    """validate key or encoded_key"""
    if isinstance(encoded_key, str):
        encoded_key = bytes(encoded_key, 'utf-8')
    if encoded_key == __c29tZSByYW5kb20gc3RyaW5n:
        return True
    return False


def report(data={}, base_url='https://api.tao.ngc.nvidia.com', timeout=10):
    """report metrics"""
    url = f'{base_url}/api/v1/metrics'
    if isinstance(data, dict):
        data = json.dumps(data)
    resp = requests.post(url, data=data, auth=('$metricstoken', __c29tZSByYW5kb20gc3RyaW5n), timeout=timeout)
    if resp.status_code == 201:
        return None
    return f'error {resp.status_code}'
