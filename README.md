# protLLM

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)

## 📝 Description

protLLM leverages the power of Large Language Models (LLMs) to revolutionize protein region classification. Built with Python, this project offers a novel approach to understanding protein structures and functions by accurately identifying and categorizing different regions within protein sequences. Unlock new insights in proteomics research with protLLM's advanced classification capabilities.

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
torch: 2.6.0+cu124
torchvision: 0.21.0+cu124
torchaudio: 2.6.0+cu124
transformers: 4.57.1
scikit-learn: 1.7.2
numpy: 2.2.6
pandas: 2.3.3
biopython: 1.86
tqdm: 4.67.1
```

## 📁 Project Structure

```
.
├── BioLiP.txt.gz
├── Model_selection.ipynb
├── fig2_roc_pocket.png
├── fig3_sequence_track.png
├── fig4_metric_summary.png
├── protein.fasta.gz
├── protein_pockets.ipynb
└── requirements.txt
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

---
*This README was generated with ❤️ by ReadmeBuddy*