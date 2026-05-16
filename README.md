# Federated Schedule: Mitigating Zombie Training in Heterogeneous Edge Networks

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Executive Summary
In distributed federated learning environments, system heterogeneity—specifically unpredictable client availability—causes massive computational waste. When edge devices drop offline before completing their assigned rigid workloads, the localized computational effort is entirely discarded. This phenomenon is known as **Zombie Training**.

**Federated Schedule** is an adaptive system architecture designed to eliminate this waste. By utilizing a proactive feedback loop and dynamically scaling assigned epochs based on predicted client availability, the system ensures optimal **Training Goodput** without sacrificing global model convergence.

## System Architecture
Unlike standard Federated Averaging (FedAvg), which assigns rigid computational workloads, Federated Schedule implements a dynamic allocation pipeline:
1. **Behavioral Prediction:** The central server anticipates individual client drop-off windows based on historical volatility (e.g., distinguishing between highly stable "Overnight Chargers" and volatile "Irregular Commuters").
2. **Adaptive Workload Scheduling:** Epoch assignments are scaled dynamically using a calculated safety buffer ($\gamma = 0.8$).
3. **Resource Optimization:** By ensuring clients only receive workloads they can realistically complete, computational cycles are preserved, and network bandwidth is optimized.

## Simulation & Results
The core simulation evaluates the architecture using a Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset across 1,000 simulated clients with severe statistical (Non-IID) and system heterogeneity. 

The framework was benchmarked against industry standards:
* **FedAvg:** Demonstrated severe vulnerability to dropouts, resulting in high Zombie Training rates.
* **FedProx:** Maintained statistical stability via a proximal term ($\mu$) but failed to address systemic computational waste.
* **Federated Schedule (Ours):** Maximized Training Goodput (Completed Updates / Initiated Updates) while maintaining competitive global test accuracy over 200 communication rounds.

## Repository Structure
```text
federated-schedule-system/
├── docs/
│   ├── project_proposal.pdf        # Initial system scope and architecture design
│   ├── final_paper.pdf             # Comprehensive research findings and goodput analysis
│   └── presentation_slides.pdf     # 16-slide academic synthesis for graduate-level audience
├── src/
│   └── federated_schedule_simulation.ipynb # PyTorch implementation and visualization
└── README.md
