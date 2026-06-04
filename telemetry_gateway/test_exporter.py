#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest
from unittest.mock import MagicMock, Mock, patch, mock_open, call

import pytest

from .exporter import AppMetrics, main


class TestAppMetrics(unittest.TestCase):
    """Test cases for AppMetrics class"""

    def test_init_default_polling_interval(self):
        """Test initialization with default polling interval"""
        app_metrics = AppMetrics()
        assert app_metrics.polling_interval_seconds == 5
        assert app_metrics.gauges == {}

    def test_init_custom_polling_interval(self):
        """Test initialization with custom polling interval"""
        app_metrics = AppMetrics(polling_interval_seconds=10)
        assert app_metrics.polling_interval_seconds == 10
        assert app_metrics.gauges == {}

    @patch('telemetry_gateway.exporter.time.sleep')
    @patch.object(AppMetrics, 'fetch')
    def test_run_metrics_loop(self, mock_fetch, mock_sleep):
        """Test the metrics loop runs fetch and sleeps correctly"""
        app_metrics = AppMetrics(polling_interval_seconds=3)
        
        # Make the loop run only twice then raise an exception to break
        mock_sleep.side_effect = [None, KeyboardInterrupt()]
        
        with pytest.raises(KeyboardInterrupt):
            app_metrics.run_metrics_loop()
        
        assert mock_fetch.call_count == 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(3)

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.Gauge')
    def test_fetch_with_get_metrics_data(self, mock_gauge_class, mock_get_metrics):
        """Test fetch when get_metrics returns data"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = {
            'metric1': 100,
            'metric2': 200,
            'last_updated': '2025-10-20',  # Should be skipped
            '_id': '123',  # Should be skipped
            'name': 'test'  # Should be skipped
        }
        
        mock_gauge1 = MagicMock()
        mock_gauge2 = MagicMock()
        mock_gauge_class.side_effect = [mock_gauge1, mock_gauge2]
        
        # Execute
        app_metrics.fetch()
        
        # Verify
        assert mock_gauge_class.call_count == 2
        mock_gauge_class.assert_any_call('metric1', 'metric1')
        mock_gauge_class.assert_any_call('metric2', 'metric2')
        mock_gauge1.set.assert_called_once_with(100)
        mock_gauge2.set.assert_called_once_with(200)
        assert 'metric1' in app_metrics.gauges
        assert 'metric2' in app_metrics.gauges

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.get_root')
    @patch('telemetry_gateway.exporter.os.path.exists')
    @patch('telemetry_gateway.exporter.Gauge')
    def test_fetch_with_metrics_file(self, mock_gauge_class, mock_exists, 
                                     mock_get_root, mock_get_metrics):
        """Test fetch when get_metrics returns None but metrics.json exists"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = None
        mock_get_root.return_value = '/app/root'
        mock_exists.return_value = True
        
        metrics_data = {'cpu_usage': 75, 'memory_usage': 85}
        mock_gauge1 = MagicMock()
        mock_gauge2 = MagicMock()
        mock_gauge_class.side_effect = [mock_gauge1, mock_gauge2]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(metrics_data))):
            # Execute
            app_metrics.fetch()
        
        # Verify
        mock_exists.assert_called_once_with('/app/root/metrics.json')
        assert mock_gauge_class.call_count == 2
        mock_gauge1.set.assert_called_once_with(75)
        mock_gauge2.set.assert_called_once_with(85)

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.get_root')
    @patch('telemetry_gateway.exporter.os.path.exists')
    def test_fetch_no_metrics_file(self, mock_exists, mock_get_root, mock_get_metrics):
        """Test fetch when get_metrics returns None and metrics.json doesn't exist"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = None
        mock_get_root.return_value = '/app/root'
        mock_exists.return_value = False
        
        # Execute
        app_metrics.fetch()
        
        # Verify - should return early without processing
        mock_exists.assert_called_once_with('/app/root/metrics.json')
        assert app_metrics.gauges == {}

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.Gauge')
    def test_fetch_reuses_existing_gauge(self, mock_gauge_class, mock_get_metrics):
        """Test that fetch reuses existing gauges instead of creating new ones"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = {'metric1': 100}
        
        mock_gauge = MagicMock()
        mock_gauge_class.return_value = mock_gauge
        
        # First fetch - creates gauge
        app_metrics.fetch()
        assert mock_gauge_class.call_count == 1
        mock_gauge.set.assert_called_with(100)
        
        # Second fetch - reuses gauge
        mock_get_metrics.return_value = {'metric1': 150}
        app_metrics.fetch()
        
        # Gauge should still be created only once
        assert mock_gauge_class.call_count == 1
        # But set should be called twice
        assert mock_gauge.set.call_count == 2
        mock_gauge.set.assert_called_with(150)

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.Gauge')
    def test_fetch_handles_exception(self, mock_gauge_class, mock_get_metrics):
        """Test that fetch handles exceptions gracefully"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = {'metric1': 100}
        mock_gauge_class.side_effect = Exception("Test exception")
        
        # Execute - should not raise exception
        app_metrics.fetch()
        
        # Verify - gauges dict should be empty since exception occurred
        assert app_metrics.gauges == {}

    @patch('telemetry_gateway.exporter.get_metrics')
    @patch('telemetry_gateway.exporter.Gauge')
    def test_fetch_converts_values_to_int(self, mock_gauge_class, mock_get_metrics):
        """Test that fetch converts metric values to integers"""
        # Setup
        app_metrics = AppMetrics()
        mock_get_metrics.return_value = {
            'metric1': '100',
            'metric2': 50.7
        }
        
        mock_gauge1 = MagicMock()
        mock_gauge2 = MagicMock()
        mock_gauge_class.side_effect = [mock_gauge1, mock_gauge2]
        
        # Execute
        app_metrics.fetch()
        
        # Verify - values are converted to int
        mock_gauge1.set.assert_called_once_with(100)
        mock_gauge2.set.assert_called_once_with(50)


class TestMain(unittest.TestCase):
    """Test cases for main function"""

    @patch('telemetry_gateway.exporter.time.sleep')
    @patch('telemetry_gateway.exporter.start_http_server')
    @patch('telemetry_gateway.exporter.AppMetrics')
    @patch('telemetry_gateway.exporter.os.getenv')
    def test_main_with_default_values(self, mock_getenv, mock_app_metrics_class,
                                     mock_start_http_server, mock_sleep):
        """Test main function with default environment values"""
        # Setup
        mock_getenv.side_effect = lambda key, default: default
        mock_app_metrics = MagicMock()
        mock_app_metrics_class.return_value = mock_app_metrics
        mock_app_metrics.run_metrics_loop.side_effect = KeyboardInterrupt()
        
        # Execute
        with pytest.raises(KeyboardInterrupt):
            main()
        
        # Verify
        mock_getenv.assert_any_call("POLLING_INTERVAL_SECONDS", "5")
        mock_getenv.assert_any_call("EXPORTER_PORT", "9877")
        mock_app_metrics_class.assert_called_once_with(polling_interval_seconds=5)
        mock_start_http_server.assert_called_once_with(addr='0.0.0.0', port=9877)
        mock_app_metrics.run_metrics_loop.assert_called_once()

    @patch('telemetry_gateway.exporter.time.sleep')
    @patch('telemetry_gateway.exporter.start_http_server')
    @patch('telemetry_gateway.exporter.AppMetrics')
    @patch('telemetry_gateway.exporter.os.getenv')
    def test_main_with_custom_env_values(self, mock_getenv, mock_app_metrics_class,
                                        mock_start_http_server, mock_sleep):
        """Test main function with custom environment values"""
        # Setup
        def getenv_side_effect(key, default):
            env_vars = {
                "POLLING_INTERVAL_SECONDS": "10",
                "EXPORTER_PORT": "8080"
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_app_metrics = MagicMock()
        mock_app_metrics_class.return_value = mock_app_metrics
        mock_app_metrics.run_metrics_loop.side_effect = KeyboardInterrupt()
        
        # Execute
        with pytest.raises(KeyboardInterrupt):
            main()
        
        # Verify
        mock_app_metrics_class.assert_called_once_with(polling_interval_seconds=10)
        mock_start_http_server.assert_called_once_with(addr='0.0.0.0', port=8080)


if __name__ == '__main__':
    unittest.main()

