# Generating Graphs with Specified Properties

This repository contains the implementation of a graph generation model for the Kaggle competition **"Generating Graphs with Specified Properties"**. The project leverages **Variational Graph Autoencoders (VGAE)** and a **denoising diffusion model** to generate graphs conditioned on textual descriptions.

## Overview

Graphs are a powerful way to represent relationships and structures. This project aims to generate graphs that align with semantic constraints extracted from textual descriptions. Key features of this implementation include:

- **VGAE** for encoding and decoding graph structures.
- Integration of **pretrained text embeddings** to enrich the graph generation process.
- A **denoising diffusion model** to refine graph structures and ensure alignment with textual descriptions.
- Optimization techniques to achieve a Mean Absolute Error (MAE) reduction from **0.8 to 0.17**.

## Dataset

The dataset for this project is structured into three components:

1. **Training Set**: 8,000 samples with graph files and textual descriptions.
2. **Validation Set**: 1,000 samples for model tuning and evaluation.
3. **Test Set**: 1,000 textual descriptions for generating corresponding graphs.

The dataset includes:
- Graph files in **GraphML** and **edgelist** formats.
- Textual descriptions containing structural and statistical graph properties.

## Code Structure

The repository is organized as follows:

### 1. `autoencoder.py`
This file contains the implementation of the **Variational Graph Autoencoder (VGAE)**, which has two main components:
- **Graph Encoder**: Converts graph adjacency matrices and node features into a latent representation (\( \mu \) and \( \sigma \)).
- **Graph Decoder**: Generates adjacency matrices from the latent representation and textual embeddings.

Key feature:
- Uses **GraphEncoder_1**, an improved graph convolutional encoder, to learn richer graph embeddings.

### 2. `denoise_model.py`
This file implements the **denoising diffusion model**, which:
- Refines latent graph representations to align them with textual embeddings.
- Incorporates pretrained text embeddings from **jinaai/jina-embeddings-v3**.
- Uses **Huber Loss** for robust noise prediction.

### 3. `utils.py`
Contains helper functions for:
- **Graph Preprocessing**: Converting graphs to adjacency matrices, computing spectral embeddings, and normalizing node features.
- **Text Preprocessing**: Using the pretrained **jinaai/jina-embeddings-v3** model to extract text embeddings.
- Data loading and formatting for use in machine learning models.

### 4. `main.py`
The main script orchestrates the training and evaluation pipeline:
- Preprocesses the dataset.
- Trains the VGAE and denoising diffusion models.
- Generates graphs for the test set based on textual descriptions.
- Outputs generated graphs as adjacency matrices.

## Methodology

### 1. **Data Preprocessing**
- Graphs are converted to adjacency matrices and standardized using **Breadth-First Search (BFS)**.
- **Spectral embeddings** (Laplacian eigenvectors) are computed for node features.
- Textual descriptions are encoded into dense vectors using **jinaai/jina-embeddings-v3**.

### 2. **Variational Graph Autoencoder (VGAE)**
- Encodes the graph structure into a latent representation (\( \mu \) and \( \sigma \)).
- Reparameterization trick generates latent vectors (\( z = \mu + \sigma \cdot \epsilon \)).
- Decodes the latent vectors, combined with text embeddings, into adjacency matrices.

### 3. **Denoising Diffusion Model**
- Refines latent representations by incorporating text embeddings.
- Uses **Huber Loss** for noise prediction to balance robustness and stability.

## Results

- **VGAE Improvements**: Reduced MAE from **0.8 to 0.3** by incorporating spectral and text-based features.
- **Denoising Model Optimization**: Text embeddings further reduced MAE to **0.17** by aligning semantic constraints with graph structures.

