# Financial Time Series Clustering

[![arXiv](https://img.shields.io/badge/arXiv-2402.11066-b31b1b.svg)](https://arxiv.org/abs/2402.11066)

## Overview

This repository contains the official implementation of "Towards Financially Inclusive Credit Products Through Financial Time Series Clustering" by Tristan Bester and Benjamin Rosman, published in AAAI W5: AI in Finance for Social Impact.

The project presents a novel time series clustering algorithm designed to help financial institutions understand consumer financial behavior through transaction data without relying on restrictive credit scoring techniques. This approach promotes financial inclusion by enabling institutions to create more tailored financial products based on actual spending behavior.

## Abstract

Financial inclusion ensures that individuals have access to financial products and services that meet their needs. As a key contributing factor to economic growth and investment opportunity, financial inclusion increases consumer spending and consequently business development. It has been shown that institutions are more profitable when they provide marginalised social groups access to financial services. 

Customer segmentation based on consumer transaction data is a well-known strategy used to promote financial inclusion. While the required data is available to modern institutions, the challenge remains that segment annotations are usually difficult and/or expensive to obtain. This prevents the usage of time series classification models for customer segmentation based on domain expert knowledge. 

As a result, clustering is an attractive alternative to partition customers into homogeneous groups based on the spending behaviour encoded within their transaction data. In this paper, we present a solution to one of the key challenges preventing modern financial institutions from providing financially inclusive credit, savings and insurance products: the inability to understand consumer financial behaviour, and hence risk, without the introduction of restrictive conventional credit scoring techniques. We present a novel time series clustering algorithm that allows institutions to understand the financial behaviour of their customers. This enables unique product offerings to be provided based on the needs of the customer, without reliance on restrictive credit practices.

## Requirements

### Environment Setup

You can set up the environment using Conda:

```bash
conda env create -f environment.yml
conda activate berka
```

Required packages include:
- PyTorch
- NumPy
- Pandas
- scikit-learn
- MongoDB Python driver
- tqdm
- python-dotenv

### Database Setup

The project uses MongoDB to store configurations and results. You can run MongoDB using Docker:

```bash
docker compose up -d
```

Configure your database credentials in a `.env` file:
```
MONGO_USERNAME=root
MONGO_PASSWORD=rootpassword
```

## Data

This project uses the Berka dataset, which contains banking transactions. To use the system, place the dataset files in the following structure:

```
data/
└── Berka/
    ├── account.csv  - Account information (4502 accounts)
    ├── card.csv     - Card details (894 cards)
    ├── client.csv   - Client information (5371 clients)
    ├── disp.csv     - Dispositions (account-client relationships)
    ├── district.csv - District/demographic data
    ├── loan.csv     - Loan information
    ├── order.csv    - Payment orders
    └── trans.csv    - Transaction data
```

## Usage

### 1. Initialize the database with configurations

```bash
python init_db.py
```

This creates a database with various model configurations to evaluate.

### 2. Run the clustering experiments

```bash
python main.py
```

This will:
1. Load the Berka dataset
2. Process financial transactions
3. Train different autoencoder architectures
4. Apply clustering methods
5. Evaluate clusters using metrics like Silhouette Score and Davies-Bouldin Index
6. Store results in the MongoDB database

## Project Structure

- `main.py`: Main script to run experiments
- `init_db.py`: Script to initialize the database with configurations
- `src/`: Source code directory
  - `datasets/`: Dataset handling classes
  - `models/`: Neural network models
  - `drivers/`: Training procedures
  - `factories/`: Factory methods for model components
  - `db/`: Database interaction
  - `modules/`: Neural network modules
- `data/`: Directory for dataset files
- `plots/`: Directory for saved visualizations
- `environment.yml`: Conda environment configuration
- `docker-compose.yml`: Docker configuration for MongoDB

## Architecture

The system implements multiple neural network architectures for financial time series clustering:
- Fully Connected Neural Networks (FCNN)
- Residual Networks (ResNet)
- Long Short-Term Memory networks (LSTM)
- Deep Temporal Clustering (DTC)

Various pretext losses are implemented:
- Mean Squared Error (MSE)
- Multi-task Reconstruction (multi_rec)
- Variational Autoencoders (VAE)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bester2024towards,
  title={Towards Financially Inclusive Credit Products Through Financial Time Series Clustering},
  author={Bester, Tristan and Rosman, Benjamin},
  journal={AAAI W5: AI in Finance for Social Impact},
  year={2024},
  eprint={2402.11066},
  archivePrefix={arXiv}
}
```