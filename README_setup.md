# TAPS-EGNN

**Topology-Aware Power System State Estimation using Graph Neural Networks**

This repository provides a Docker-based environment for development, training, and experimentation with GNN models for power system state estimation.

## Set-up docker environment 
### 1. Clone the repository
```bash
git clone https://github.com/prajapati-incontrol/TapSEGNN
cd tapsegnn
```

### 2. Pre-pull PyTorch base image (optional)
```bash
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
```

### 3. Build Docker images
```bash
docker-compose build
```


## Use Docker services

### Development (Jupyter Lab)
```bash
docker-compose up dev
```
Access Jupyter at: http://localhost:8888

Token: dev

Volumes:

./src → /app/src

./data → /app/data

./results → /app/results


### Train GAN's
```bash
docker-compose up train
```
Runs main.py with config/config.yaml

Use custom config:
    CONFIG_FILE=config/config_custom.yaml docker-compose up train


