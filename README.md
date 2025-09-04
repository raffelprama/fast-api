# FastAPI LLM Benchmark API

<p align="center">
  <img src="assets/logo.svg" alt="Project Logo" width="120" />
</p>

<p align="center">
  <strong>FastAPI service for benchmarking LLM endpoints: latency, throughput, and errors</strong>
  <br/>
  <a href="#getting-started">Getting Started</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#load-testing">Load Testing</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#license">License</a>
  <br/>
  <a href="https://fastapi.tiangolo.com/">FastAPI</a> ·
  <a href="https://uvicorn.org/">Uvicorn</a> ·
  <a href="https://kubernetes.io/">Kubernetes</a> ·
  <a href="https://locust.io/">Locust</a>
</p>

<p align="center">
  <a href="https://ghcr.io/raffelprama/my-fastapi"><img src="https://img.shields.io/badge/image-ghcr.io%2Fraffelprama%2Fmy--fastapi-blue" alt="GHCR Image" /></a>
  <a href="https://hub.docker.com/"><img src="https://img.shields.io/badge/runtime-uvicorn%2Bgunicorn-green" alt="Runtime" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="License: MIT" /></a>
</p>

## Overview

This repository contains a FastAPI application tailored for benchmarking Large Language Model (LLM) APIs and services. It includes:

- A production-ready Dockerfile
- Kubernetes manifests (`deployment.yaml`, `service.yaml`)
- A `locustfile.py` for load testing

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Setup (Local)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://localhost:8000/docs`

## Configuration

Environment variables are provided via Kubernetes Secret created from a `.env` file. To create/update the Secret:

```bash
kubectl create secret generic bechmark-beta-api-env --from-env-file=.env -n benchmark --dry-run=client -o yaml | kubectl apply -f -
```

The deployment consumes it via `envFrom.secretRef` in `deployment.yaml`.

## Docker

Build and run locally:

```bash
docker build -t my-fastapi:local .
docker run -p 8000:8000 --env-file .env my-fastapi:local
```

## Deployment

Apply Kubernetes manifests:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Ensure image pull secret `regcred` exists in namespace `benchmark` if you use a private registry.

## Load Testing

This repo includes `locustfile.py` with tasks suited for LLM-style requests (prompt/response), enabling RPS, latency, P95/P99, and error-rate analysis.

Run Locust locally:

```bash
locust -f locustfile.py
```

Then open `http://localhost:8089` and set the host to your service URL.

## Project Structure

```
.
├─ main.py            # FastAPI app
├─ locustfile.py      # Locust load tests
├─ Dockerfile         # Production build
├─ deployment.yaml    # Kubernetes deployment
├─ service.yaml       # Kubernetes service
├─ requirements.txt   # Python dependencies
└─ assets/logo.svg    # Logo (replace as needed)
```

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).


