# bitcoin-otc-trust-gat
## Overview
A Graph Attention Network for predicting trust vs. distrust relationships between traders on the Bitcoin OTC platform. Uses PyTorch Geometric to learn node embeddings via multi-head attention over the trust network, then predicts whether a given trading relationship is trustworthy — with explicit attention to the class imbalance problem inherent in real-world trust networks.

## Dataset
- **Source:** [Bitcoin OTC Trust Weighted Signed Network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) (Stanford SNAP)
- ~5,900 trust ratings between ~3,700 Bitcoin traders, rated -10 (total distrust) to +10 (total trust)
- Task framed as binary edge classification: **Trust** (rating > 0) vs. **Distrust** (rating ≤ 0)
- Highly imbalanced: ~90% of edges are trust, ~10% are distrust

## Approach
**Node features:** in-degree, out-degree, average rating given, average rating received — normalized per node.

**Architecture — 2-layer Graph Attention Network:**
- Layer 1: `GATConv` with 4 attention heads, learning node embeddings by attending over each node's trading neighborhood
- Layer 2: `GATConv` with single-head output, producing final node embeddings
- Edge-level prediction: concatenated source/target node embeddings passed through a small MLP to predict trust vs. distrust

**Handling class imbalance:** An initial unweighted model achieved a misleadingly high overall F1 (0.96) while only catching 45% of actual distrust relationships — the model was effectively defaulting toward the majority class. Applied class-weighted `BCEWithLogitsLoss` (upweighting the minority Distrust class ~9x based on the train-set class ratio) to explicitly penalize missed distrust cases.

**Attention analysis:** Extracted and ranked attention weights from the first GAT layer to identify the most influential trust relationships in the network. Self-loops (added automatically by `GATConv`) were filtered out of this ranking, since they trivially receive attention = 1.0 and aren't meaningful signal.

## Results


|
 Metric 
|
 Unweighted Loss 
|
 Class-Weighted Loss 
|
|
--------
|
------------------
|
----------------------
|
|
 AUC 
|
 0.867 
|
**
0.911
**
|
|
 Distrust Recall 
|
 0.45 
|
**
0.79
**
|
|
 Distrust Precision 
|
 0.70 
|
 0.45 
|

Final classification report (class-weighted model):


|
 Class 
|
 Precision 
|
 Recall 
|
 F1 
|
|
-------
|
-----------
|
--------
|
-----
|
|
 Trust 
|
 0.97 
|
 0.89 
|
 0.93 
|
|
 Distrust 
|
 0.45 
|
 0.79 
|
 0.57 
|

Overall accuracy: 0.88 | AUC: 0.911

**Note on the tradeoff:** Class weighting traded distrust precision for a large recall gain (45%→79%). In a fraud-relevant task like this, missing a genuine bad actor (false negative) is typically more costly than flagging a legitimate trader for review (false positive) — so this tradeoff is a deliberate, defensible choice rather than a limitation.

## What I'd Improve Next
- Try focal loss as an alternative to class-weighted BCE, which can handle imbalance without sacrificing as much precision
- Incorporate edge timestamps (available in the raw data but unused here) to model how trust evolves over time
- Compare against a GCN or GraphSAGE baseline to quantify the specific benefit of attention over uniform neighbor aggregation

## Tech Stack
PyTorch, PyTorch Geometric, scikit-learn, Pandas, NumPy

## Project Structure
```
├── notebook.ipynb          # Full pipeline: data loading, GAT model, training, evaluation, attention analysis
├── requirements.txt
└── README.md
```
