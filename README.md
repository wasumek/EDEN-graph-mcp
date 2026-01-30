# EDEN-Graph: Network Analysis & Market Clustering Module

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5220%2F0013359800003911-blue)](https://doi.org/10.5220/0013359800003911)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Scientific Abstract
**EDEN-Graph** is the open-source reference implementation of the research presented at **BIOSTEC 2025**. While the original paper establishes the theoretical framework for **Network Analysis** in healthy longevity ecosystems, this repository extends that work by introducing a novel **Market Clustering** module.

By integrating **Natural Language Processing (NLP)** (via NMF) with **Hierarchical Clustering**, EDEN-Graph provides a dual-layer approach to incentive alignment:
1.  **Network Topology**: Mapping stakeholder alignment via value proposition similarity (as detailed in the paper).
2.  **Market Segmentation**: A new extension for profiling ventures against financial and structural metrics.

## Methodological Framework
The codebase implements the **Modeling Phase** of a modified CRISP-DM process:

### 1. Unstructured Data Processing (`app.py`)
Value propositions are treated as high-dimensional test vectors. The pipeline applies:
*   **Tokenization & Lemmatization**: NLTK-based normalization of industry vernacular.
*   **TF-IDF Vectorization**: Statistical weighting of term importance across the corpus.

### 2. Topic Modeling (NMF)
The framework utilizes **Non-negative Matrix Factorization (NMF)** to decompose the document-term matrix into latent semantic topics (\(k=8\)). This dimensionality reduction allows for the identification of thematic clusters within the ecosystem (e.g., "Remote Monitoring", "Behavioral Interventions").

### 3. Network Topology Construction
Stakeholder alignment is modeled as a weighted undirected graph \(G=(V, E)\):
*   **Nodes (\(V\))**: Market segments or individual organizations.
*   **Edges (\(E\))**: Defined by cosine similarity \(S_C(A, B)\) between topic vectors.
*   **Thresholding**: Edges are constructed only where \(S_C > P_{50}\) (50th percentile) to highlight significant strategic alignment.

## Repository Structure
*   `app.py`: Core computational engine (Flask).
*   `market_analyzer.py`: Hierarchical clustering module for market segmentation.
*   `0_Longevity World.xlsx`: Curated dataset of 128 stakeholder value propositions.
*   `templates/`: Jinja2 templates for the clean academic interface.

## Data Privacy & Anonymization
Although the underlying datasets (`0_Longevity World.xlsx`, `DHT_T2D...`) contain publicly available information from Crunchbase and official company websites, this framework **programmatically anonymizes** all organization names during runtime.
*   Organizations are rendered as generic labels (e.g., "Insurer 4", "Digital Therapeutics 12").
*   This feature is implemented to ensure neutrality and prevent potential conflict of interest during the research analysis.

## Reproduction Instructions

### Prerequisites
*   Python 3.10+
*   Virtual Environment (Standard scientific stack)

### Setup
1.  **Clone and Install**:
    ```bash
    git clone https://github.com/wasumek/EDEN-graph-mcp.git
    cd EDEN-graph-mcp
    pip install -r requirements.txt
    ```
2.  **Initialize NLP Resources**:
    Run the setup script to download specific NLTK corpora (`stopwords`, `wordnet`, `punkt_tab`):
    ```bash
    python setup_nltk.py
    ```
3.  **Environment Configuration**:
    Create a `.env` file for the OpenAI API (used for the Generative Interpretation module):
    ```
    OPENAI_API_KEY=sk-...
    ```

### Execution
Launch the local analysis server:
```bash
python app.py
```
> Access the interface at `http://127.0.0.1:5001`.

## Citation
Please cite the accompanying conference paper for the core **Network Analysis** framework. If you utilize the **Market Clustering** extension, please acknowledge this repository directly.

```bibtex
@conference{eden2025,
    author={Wasu Mekniran and Tobias Kowatsch},
    title={EDEN: Towards a Computational Framework to Align Incentives in Healthy Aging},
    booktitle={Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies - Vol 2: HEALTHINF},
    year={2025},
    pages={1067-1076},
    publisher={ScitePress},
    organization={INSTICC},
    doi={10.5220/0013359800003911},
    isbn={978-989-758-742-6}
}
```

## License
*   **Source Code**: MIT License (Open Source)
*   **Methodology & Concepts**: CC BY-NC-ND 4.0 (for academic attribution)

---
*Developed at the Centre for Digital Health Interventions (CDHI), ETH Zurich & University of St. Gallen.*
