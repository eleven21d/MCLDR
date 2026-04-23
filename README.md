# MCLDR

PyTorch implementation of:

**Cross-View Contrastive Learning with Closeness-Centrality-Aware Structural Denoising for Course Recommendation**

## Overview
MCLDR is a cross-view contrastive learning framework for course recommendation on heterogeneous information networks (HINs).  
It jointly models:

- **Meta-path semantic view** for high-order relation modeling  
- **Denoised structural view** for reliable structure learning  
- **Cross-view contrastive learning** for representation alignment  

to address semantic inconsistency and structural noise under sparse supervision.

## Environment
- Python 3.10  
- PyTorch 2.0  
-  DGL 1.1.2 (CUDA 11.8)

## Dataset
- **MOOCCube**
- **Amazon (Product)**  
Get the datasets from https://drive.google.com/drive/folders/1Yx2-q2yhO629IVaZsjhBzbhmd6QR3s1G?usp=drive_link

Data includes interaction data, meta-path graphs, and denoised structural graphs.

## Generate Denoised Graph
```bash
python model/denoising_amazon.py --data_dir data/mooc --out_dir data/mooc_denoised --beta 0.03 --cold_deg_thresh 2
```
then Loda in `main.py` 
```python
denoise_path = "data/mooc_denoised/G_denoised_beta0.03.txt"
```
## Run the code
```bash
python main.py --dataset mooc --data_path data/ --dim 64 --temperature 0.2 --lr 0.001 --epochs 100
```

## Citation

```bibtex
@inproceedings{MCLDRxxxx,
  author = {Wei Zhang, Shiyi Zhu∗, Xinyao Zeng and Yu Zhang},
  title = {{Cross-View Contrastive Learning with Closeness-Centrality-Aware Structural Denoising for Course Recommendation}},
  booktitle = {Underreview},
  year = {2026},
}
```

## Related Papers
IHGCL:Intent-Guided Heterogeneous Graph Contrastive Learning for Recommendation
https://github.com/wangyu0627/IHGCL

KGAN: Knowledge Grouping Aggregation Network for Course Recommendation in MOOCs
https://github.com/StZHY/KGAN

BHGCL: Bottleneced Heterogeneous Graph Contrastive Learning for Robust Recommendation
https://github.com/DuellingSword/BHGCL

ACKRec:Attentional Graph Convolutional Networks for Knowledge Concept Recommendation in MOOCs in a Heterogeneous View
https://github.com/JockWang/ACKRec

HeCo:Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning
https://github.com/BUPT-GAMMA/HeCo

SSLRec: A Self-Supervised Learning Framework for Recommendation
https://github.com/HKUDS/SSLRec
