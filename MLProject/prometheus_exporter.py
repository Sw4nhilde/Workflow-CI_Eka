from prometheus_client import Counter, Histogram, Gauge, Info, Summary, start_http_server
import time
import random
import threading

# Create metrics (10 different metrics as required)
REQUESTS = Counter(
    "requests_total",
    "Total number of requests"
)

ERRORS = Counter(
    "errors_total",
    "Total number of errors"
)

LATENCY = Histogram(
    "latency_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTIONS = Counter(
    "predictions_total",
    "Total predictions made"
)

CHURN_PREDICTED = Counter(
    "churn_predicted_total",
    "Total churn predictions"
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Currently active requests"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy score"
)

CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent", 
    "Memory usage percentage"
)

REQUEST_SIZE = Summary(
    "request_size_bytes",
    "Request size in bytes"
)

# Simulate metrics updates
def update_metrics():
    while True:
        time.sleep(5)
        # Simulate random metrics
        CPU_USAGE.set(random.uniform(10, 90))
        MEMORY_USAGE.set(random.uniform(20, 80))
        MODEL_ACCURACY.set(random.uniform(0.75, 0.95))
        
        # Random increments
        REQUESTS.inc(random.randint(0, 10))
        if random.random() < 0.1:  # 10% error rate
            ERRORS.inc()
        
        PREDICTIONS.inc(random.randint(0, 5))
        if random.random() < 0.3:  # 30% churn rate
            CHURN_PREDICTED.inc()

# Start metrics update thread
threading.Thread(target=update_metrics, daemon=True).start()

# Start HTTP server
start_http_server(8000)
print("Prometheus exporter running on http://localhost:8000")
print("Metrics available at http://localhost:8000/metrics")

# Keep the script running
try:
    while True:
        time.sleep(1)
        # Simulate active requests
        active = random.randint(0, 20)
        ACTIVE_REQUESTS.set(active)
        
        # Simulate request latency
        with LATENCY.time():
            time.sleep(random.uniform(0.01, 0.5))
        
        # Simulate request size
        REQUEST_SIZE.observe(random.randint(100, 5000))
        
except KeyboardInterrupt:
    print("\nExporter stopped")