#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import re
import time
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_metrics, get_root
from prometheus_client import start_http_server, Gauge, Counter


class AppMetrics:
    """Enhanced Prometheus metrics exporter supporting both flat and labeled metrics.
    
    Handles two metric formats:
    1. Flat metrics (legacy): simple key-value pairs
       Example: total_action_train_pass = 123
    
    2. Labeled metrics (new): Prometheus-style with labels
       Example: tao_job_total{network="resnet50",action="train",...} = 5
    
    Correctly uses Counter vs Gauge based on metric naming conventions.
    """

    def __init__(self, polling_interval_seconds=5):
        self.polling_interval_seconds = polling_interval_seconds

        # OLD: Prometheus metrics for flat format (backward compatibility)
        self.gauges = {}
        
        # NEW: Prometheus metrics for labeled format
        self.labeled_metrics = {}  # {metric_name: prometheus_metric_object}
        self._previous_counter_values = {}  # Track counter deltas

    def run_metrics_loop(self):
        """Metrics fetching loop"""

        while True:
            self.fetch()
            time.sleep(self.polling_interval_seconds)

    def fetch(self):
        """Get metrics from application and refresh Prometheus metrics with new values."""
        status_data = get_metrics()
        if not status_data:
            metrics_file_path = os.path.join(get_root(), 'metrics.json')
            if not os.path.exists(metrics_file_path):
                print('Metrics.json file does not exist. Skip fetch.')
                return
            with open(metrics_file_path, 'r') as f:
                status_data = json.load(f)
        
        print()
        print("****** Fetching metrics ******")
        print()

        # Fetch raw status data from the application
        try:
            # Update Prometheus metrics with application metrics
            for k, v in status_data.items():
                if k in ['last_updated', '_id', 'name']:
                    continue

                # Detect metric format
                if '{' in k and '}' in k:
                    # NEW FORMAT: Labeled metric
                    self._handle_labeled_metric(k, v)
                else:
                    # OLD FORMAT: Flat metric (backward compatibility)
                    self._handle_flat_metric(k, v)
                    
        except Exception as e:
            print(f"Error in fetch: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_flat_metric(self, key, value):
        """Handle old flat metrics (backward compatibility).
        
        Args:
            key: Metric key (e.g., 'total_action_train_pass')
            value: Metric value (int)
        """
        gauge = self.gauges.get(key)
        if gauge is None:
            print(f'Creating new gauge for key: {key}')
            self.gauges[key] = Gauge(key, key)
        gauge = self.gauges[key]
        gauge.set(int(value))
        print(f'{key}: {value}')
    
    def _handle_labeled_metric(self, metric_key, value):
        """Handle Prometheus-style labeled metrics.
        
        Args:
            metric_key: Labeled metric key (e.g., 'tao_job_total{action="train",network="resnet50"}')
            value: Metric value (int)
        """
        # Parse metric key
        metric_name, labels = self._parse_labeled_metric(metric_key)
        
        if not metric_name:
            print(f'Warning: Could not parse labeled metric: {metric_key}')
            return
        
        # Determine metric type from name
        metric_type = self._infer_metric_type(metric_name)
        
        # Create metric if doesn't exist
        if metric_name not in self.labeled_metrics:
            print(f'Creating new {metric_type} for: {metric_name}')
            self.labeled_metrics[metric_name] = self._create_prometheus_metric(
                metric_name,
                metric_type,
                list(labels.keys())
            )
        
        # Update metric value based on type
        if metric_type == 'counter':
            self._update_counter(metric_key, metric_name, labels, value)
        else:  # gauge
            self._update_gauge(metric_name, labels, value)
        
        print(f'{metric_key}: {value}')
    
    def _update_counter(self, metric_key, metric_name, labels, value):
        """Update a counter metric by tracking delta.
        
        Counters should only increase, so we track the delta from last value.
        
        Args:
            metric_key: Full metric key with labels
            metric_name: Base metric name
            labels: Label dictionary
            value: Current value from MongoDB
        """
        previous = self._previous_counter_values.get(metric_key, 0)
        current = int(value)
        delta = current - previous
        
        if delta > 0:
            # Normal increment
            self.labeled_metrics[metric_name].labels(**labels).inc(delta)
        elif delta < 0:
            # Counter reset (e.g., MongoDB cleared, new day)
            print(f'   Counter reset detected for {metric_key}: {previous} → {current}')
            self.labeled_metrics[metric_name].labels(**labels).inc(current)
        # else: delta == 0, no change needed
        
        self._previous_counter_values[metric_key] = current
    
    def _update_gauge(self, metric_name, labels, value):
        """Update a gauge metric (simple set).
        
        Args:
            metric_name: Base metric name
            labels: Label dictionary
            value: Current value
        """
        self.labeled_metrics[metric_name].labels(**labels).set(int(value))
    
    def _parse_labeled_metric(self, metric_key):
        """Parse Prometheus-style metric key.
        
        Args:
            metric_key: 'tao_job_total{action="train",network="resnet50"}'
            
        Returns:
            Tuple of (metric_name, labels_dict)
            Example: ('tao_job_total', {'action': 'train', 'network': 'resnet50'})
        """
        # Pattern: metric_name{label1="value1",label2="value2"}
        match = re.match(r'([^{]+)\{([^}]+)\}', metric_key)
        if not match:
            return None, {}
        
        metric_name = match.group(1)
        labels_str = match.group(2)
        
        # Parse labels: key1="value1",key2="value2"
        labels = {}
        for label_pair in labels_str.split(','):
            if '=' in label_pair:
                key, value = label_pair.split('=', 1)
                # Remove quotes from value
                value = value.strip('"').strip("'")
                labels[key.strip()] = value
        
        return metric_name, labels
    
    def _infer_metric_type(self, metric_name):
        """Infer Prometheus metric type from name following naming conventions.
        
        Args:
            metric_name: Base metric name (e.g., 'tao_job_total', 'tao_job_duration_sum')
            
        Returns:
            'counter' or 'gauge'
        """
        # Prometheus naming conventions:
        # - Counters: end with _total or _sum (monotonically increasing)
        # - Gauges: everything else (can go up and down)
        if metric_name.endswith('_total') or metric_name.endswith('_sum'):
            return 'counter'
        else:
            return 'gauge'
    
    def _create_prometheus_metric(self, metric_name, metric_type, label_names):
        """Create appropriate Prometheus metric object.
        
        Args:
            metric_name: Base metric name
            metric_type: 'counter' or 'gauge'
            label_names: List of label keys
            
        Returns:
            Prometheus metric object (Counter or Gauge)
        """
        # Generate help text from metric name
        help_text = f'TAO {metric_name.replace("_", " ")}'
        
        if metric_type == 'counter':
            return Counter(metric_name, help_text, label_names)
        else:  # gauge
            return Gauge(metric_name, help_text, label_names)


def main():
    """Main entry point"""

    polling_interval_seconds = int(os.getenv("POLLING_INTERVAL_SECONDS", "5"))
    exporter_port = int(os.getenv("EXPORTER_PORT", "9877"))

    app_metrics = AppMetrics(
        polling_interval_seconds=polling_interval_seconds
    )
    start_http_server(addr='0.0.0.0', port=exporter_port)
    app_metrics.run_metrics_loop()

if __name__ == "__main__":
    time.sleep(30)
    main()
