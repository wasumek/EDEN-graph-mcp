# EDEN-Graph: Computational Framework for Incentive Alignment in Healthy Aging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5220%2F0013359800003911-blue)](https://doi.org/10.5220/0013359800003911)
[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://doi.org/10.17605/OSF.IO/Q3EYS)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## OSF Registration Metadata

| Field                        | Details                                                     |
| :--------------------------- | :---------------------------------------------------------- |
| **Registration DOI**   | [10.17605/OSF.IO/Q3EYS](https://doi.org/10.17605/OSF.IO/Q3EYS) |
| **Registration Type**  | Simulation Study Template                                   |
| **Date Registered**    | Jan 30, 2026                                                |
| **Associated Project** | [osf.io/4nz6v](https://osf.io/4nz6v)                           |

> **Description**: This Open Science Framework (OSF) project serves as the immutable data and code repository for the software "EDEN-Graph". It includes the reference implementation for the paper "EDEN: Towards a Computational Framework to Align Incentives in Healthy Aging" (BIOSTEC 2025) and extends it with a new Market Clustering module.

## Scientific Abstract

**EDEN-Graph** is the open-source reference implementation of the research presented at **BIOSTEC 2025**. While the original paper establishes the theoretical framework for **Network Analysis** in healthy longevity ecosystems, this repository extends that work by introducing a novel **Market Clustering** module.

By integrating **Natural Language Processing (NLP)** (via NMF) with **Hierarchical Clustering**, EDEN-Graph provides a dual-layer approach to incentive alignment:

1. **Network Topology**: Mapping stakeholder alignment via value proposition similarity (as detailed in the paper).
2. **Market Segmentation**: A new extension for profiling ventures against financial and structural metrics.

## Aims & Hypotheses

**Type:** Method Evaluation / Estimation

The specific aim of this study is to evaluate the properties of **Non-negative Matrix Factorization (NMF)** and **Hierarchical Agglomerative Clustering** with respect to the task of **estimating** the latent incentive structure of the healthy longevity ecosystem.

### Research Questions

* **RQ1 (Validity)**: *To what extent can latent semantic analysis (specifically NMF with $k=8$) effectively decompose high-dimensional industry vernacular into interpretable, distinct stakeholder clusters?*
* **RQ2 (Topology)**: *Does the constructed stakeholder network reveal statistically significant "structural holes" between the "Financial/Payer" sector and the "Preventative Health" sector?*

### Hypotheses

* **H1 (Fragmentation)**: We hypothesize that the ecosystem is characterized by **semantic fragmentation**, where key stakeholders use orthogonal vocabularies resulting in low cosine similarity.
* **H2 (Recovery)**: We predict that **EDEN-Graph** will successfully recover expert-validated market segments solely from unstructured text data.

## Methodological Framework

The codebase implements the **Modeling Phase** of a modified CRISP-DM process.

### 1. Data-Generating Mechanism & Factors

In this computational experiment, we varied the following factors to determine the optimal model configuration:

* **Latent Space Dimensionality ($k$)**: Varied $k \in \{5, 8, 10, 15\}$. Selected **$k=8$** for minimum reconstruction error and maximum interpretability.
* **Edge Density Threshold ($\theta$)**: Varied cutoff to isolate significant alignments. Selected **$P_{50}$** (median).
* **Vectorization**: Compared TF-IDF vs Binary Count. Selected **TF-IDF** to penalize generic industry buzzwords.

### 2. Estimands (Targets of Interest)

* **Primary Estimand**: **Semantic Cluster Coherence**. Can the algorithm consistently recover distinct market sectors (e.g., "Clinical Trials" vs "Thematic Apps")?
* **Secondary Estimand**: **Inter-Sector Alignment Score**. The quantified semantic overlap ($0 \le S_c \le 1$) between "Payer" and "Provider" clusters, serving as a proxy for the incentive gap.

### 3. Core Algorithms

* **Unstructured Data Processing**: Tokenization & Lemmatization via NLTK.
* **Topic Modeling**: Non-negative Matrix Factorization (NMF) to decompose the document-term matrix.
* **Network Construction**: Weighted undirected graph based on Cosine Similarity.

## Repository Structure

* `app.py`: Core computational engine (Flask).
* `market_analyzer.py`: Hierarchical clustering module.
* `0_Longevity World.xlsx`: Curated dataset of 128 stakeholder value propositions.
* `templates/`: Jinja2 templates for the interface.

## Data Privacy

This framework **programmatically anonymizes** all organization names during runtime (e.g., "Insurer 4") to ensure neutrality, despite using public crunchbase data.

## Reproducibility & Environment

### Statistical Software

* **Python**: 3.12.7
* **Core**: `scikit-learn` (v1.5.2), `numpy` (v2.1.2), `pandas` (v2.2.3), `networkx` (v3.4.2)
* **NLP**: `nltk` (v3.9.1)
* **Visualization**: `plotly` (v5.24.1), `matplotlib` (v3.9.2)

### Computational Environment

Benchmarks performed on macOS Sequoia (15.x) / Linux (Heroku). Requires min 8GB RAM.

### Reproducibility Strategy

1. **Frozen Snapshots (OSF)**: Immutable code/data snapshot hosted at [10.17605/OSF.IO/Q3EYS](https://doi.org/10.17605/OSF.IO/Q3EYS).
2. **Version Control**: Active dev at [GitHub](https://github.com/wasumek/EDEN-graph-mcp).
3. **Strict Pinning**: Dependencies locked in `requirements.txt`.

## Usage Instructions

### Setup

1. **Clone**: `git clone https://github.com/wasumek/EDEN-graph-mcp.git`
2. **Install**: `pip install -r requirements.txt`
3. **NLP Init**: `python setup_nltk.py`
4. **Config**: Create `.env` with `OPENAI_API_KEY` (optional for generative features).

### Execution

Launch the local server:

```bash
python app.py
```

Access at `http://127.0.0.1:5001`.

## Citation

Please cite the accompanying conference paper:

```bibtex
@conference{eden2025,
    author={Wasu Mekniran and Tobias Kowatsch},
    title={EDEN: Towards a Computational Framework to Align Incentives in Healthy Aging},
    booktitle={Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies - Vol 2: HEALTHINF},
    year={2025},
    pages={1067-1076},
    doi={10.5220/0013359800003911}
}
```

## License

* **Code**: MIT License
* **Methodology**: CC BY-NC-ND 4.0

---

*Developed at the Centre for Digital Health Interventions (CDHI), ETH Zurich & University of St. Gallen.*
