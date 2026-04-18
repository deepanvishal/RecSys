# Serko RecSys — Technical Report

## 1. Executive Summary

## 2. Problem Framing & Business Context

## 3. Dataset & Exploratory Analysis

### 3.1 Source & Schema

### 3.2 Sparsity, Popularity, and Cohort Distribution

### 3.3 Simulation Strategy (cold / sparse / warm cohorts)

## 4. System Architecture

### 4.1 Tiered Recommender Design

### 4.2 Data Flow

### 4.3 Configuration & Reproducibility Contracts

## 5. Modeling Approaches

### 5.1 Collaborative Filtering (Tier: warm baseline)

### 5.2 Matrix Factorization — SVD (Tier: sparse)

### 5.3 Two-Tower Neural Model (Tier: warm)

### 5.4 Trending / Popularity Fallback (Tier: cold)

## 6. Training & Experiment Tracking

### 6.1 W&B Project Setup

### 6.2 Hyperparameter Sweeps

### 6.3 Compute & Performance Notes

## 7. Inference Engine

### 7.1 Tier Routing

### 7.2 Reranking & Diversity

### 7.3 Incremental Update Path

## 8. Evaluation

### 8.1 Offline Metrics (Recall@K, NDCG@K, MAP@K, Coverage)

### 8.2 Cohort Breakdowns

### 8.3 Bias Audit (popularity head/torso/tail)

### 8.4 Latency & Throughput

## 9. Serving — API & UI

### 9.1 FastAPI Endpoints

### 9.2 Streamlit Demo

## 10. Limitations & Future Work

## 11. Appendix

### A. Repository Map

### B. Run Instructions

### C. References
