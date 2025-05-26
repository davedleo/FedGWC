# FedGWC

Code for our ICML 2025 paper: **Interaction-Aware Gaussian Weighting for Clustered Federated Learning**  
Alessandro Licciardi*, Davide Leo*, Eros Fanì, Barbara Caputo, Marco Ciccone

## Abstract

Federated Learning (FL) emerged as a decentralized paradigm to train models while preserving privacy. However, conventional FL struggles with data heterogeneity and class imbalance, which degrade model performance. Clustered FL balances personalization and decentralized training by grouping clients with analogous data distributions, enabling improved accuracy while adhering to privacy constraints. This approach effectively mitigates the adverse impact of heterogeneity in FL. In this work, we propose a novel clustered FL method, FedGWC (Federated Gaussian Weighting Clustering), which groups clients based on their data distribution, allowing training of a more robust and personalized model on the identified clusters. FedGWC identifies homogeneous clusters by transforming individual empirical losses to model client interactions with a Gaussian reward mechanism. Additionally, we introduce the Wasserstein Adjusted Score, a new clustering metric for FL to evaluate cluster cohesion with respect to the individual class distribution. Our experiments on benchmark datasets show that FedGWC outperforms existing FL algorithms in cluster quality and classification accuracy, validating the efficacy of our approach.

## Paper

https://arxiv.org/pdf/2502.03340

## Usage

```
pip install -r requirements.txt
python3 datasets/CIFAR100/setup.py
python3 main.py 
```
