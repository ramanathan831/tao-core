# tao-toolkit-cloud-telemetry-gateway

Cloud service enabling all TAO Toolkit containers running on customer premises to anonymously report Telemetry metrics to NVIDIA's Kratos dashboards

### Build and Push image

```
cd tao-core \
docker build -t nvcr.io/nvstaging/tao/tao-telemetry-exporter:$USER -f telemetry_gateway/Dockerfile . \
```

Note: Please be aware of `latest` tag. Latest will affect production.


### Local deployment

To deploy k8s cluster, please following [TAO API Getting Started](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=IVA&title=TAO+API+getting+started+-+new+users)

You will need to install podmonitor crd on your local cluster. Please refer to [official doc](https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/user-guides/getting-started.md)


```
CUSTOMER CONTAINERS
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Container 1 │  │  Container 2 │  │  Container N │
│              │  │              │  │              │
│ TAO Job runs │  │ TAO Job runs │  │ TAO Job runs │
│ telemetry.py │  │ telemetry.py │  │ telemetry.py │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                  │                  │
        │ POST /api/v1/metrics                │
        │ (telemetry data)                    │
        └─────────────┬───────────────────────┘
                      ▼
        ┌─────────────────────────────-┐
        │     REMOTE SERVER            │
        │  ┌────────────────────────┐  │
        │  │   Flask API (app.py)   │  │
        │  │                        │  │
        │  │  /api/v1/metrics       │  │
        │  │  - Validates schema    │  │
        │  │  - MetricProcessor     │  │
        │  │  - Builds metrics:     │  │
        │  │    • Legacy            │  │
        │  │    • Comprehensive     │  │
        │  │    • Time              │  │
        │  └──────────┬─────────────┘  │
        │             │ save           │
        │             ▼                │
        │  ┌────────────────────────┐  │
        │  │  MongoDB Storage       │  │
        │  │                        │  │
        │  │  metrics.json          │  │
        │  │  {                     │  │
        │  │    "network_...": 123, │  │
        │  │    "total_...": 456,   │  │
        │  │    "last_updated": ... │  │
        │  │  }                     │  │
        │  └──────────▲─────────────┘  │
        │             │ continuous     │
        │             │ fetch (5s)     │
        │  ┌──────────┴─────────────┐  │
        │  │ Prometheus Exporter    │  │
        │  │ (exporter.py)          │  │
        │  │                        │  │
        │  │ AppMetrics class       │  │
        │  │ - Polls MongoDB        │  │
        │  │ - Creates Gauges       │  │
        │  │ - Exposes :9877        │  │
        │  └──────────┬─────────────┘  │
        └─────────────┼────────────────┘
                      │ scrape
                      ▼
        ┌─────────────────────────────-┐
        │   Kubernetes Monitoring      │
        │  ┌────────────────────────┐  │
        │  │   PodMonitor           │  │
        │  │   - Scrapes :9877      │  │
        │  └──────────┬─────────────┘  │
        │             ▼                │
        │  ┌────────────────────────┐  │
        │  │ kube-prometheus-stack  │  │
        │  │ - Collects metrics     │  │
        │  │ - Processes            │  │
        │  └──────────┬─────────────┘  │
        │             │ remote_write   │
        └─────────────┼────────────────┘
                      ▼
        ┌─────────────────────────────┐
        │   Remote Prometheus/        │
        │   Grafana (Observability)   │
        └─────────────────────────────┘
```
