# Bug Report Classification with RoBERTa

## Overview

This repository contains the implementation of a fine-tuned RoBERTa model for classifying performance-related bug reports in software projects. The tool significantly outperforms traditional approaches like Naive Bayes with TF-IDF by leveraging RoBERTa's deep contextual understanding of language.

This project is part of the Intelligent Software Engineering (ISE) coursework, focusing on building an intelligent tool that can automatically identify performance-related issues in bug reports from various deep learning frameworks.

## Research Paper

The implementation is based on the research described in "Fine-Tuning RoBERTa for Performance-Related Bug Report Classification," which demonstrates substantial improvements over baseline methods:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Naive Bayes + TF-IDF | 0.5747 | 0.6082 | 0.7066 | 0.5240 | 0.7066 |
| RoBERTa (Our Model) | 0.9155 | 0.8210 | 0.8297 | 0.8196 | 0.9202 |
| Improvement | +59.30% | +34.99% | +17.42% | +56.41% | +30.23% |

## Repository Structure

```
ISE-Coursework/
├── ISE.ipynb              # Main Google Colab notebook with the implementation
├── ISE PDF.pdf            # Research paper
├── data/                  # Directory containing dataset files
├── results/               # Directory containing output results
├── requirements.pdf       # Dependencies and requirements documentation
├── manual.pdf             # User manual
├── replication.pdf        # Guide to replicate the results
└── README.md              # This file
```

**Note:** The `models/` directory is not included in the repository due to file size limitations. It will be created automatically when running the code, and trained models will be saved there.

## Datasets

The implementation uses bug report datasets from five deep learning frameworks:
- TensorFlow
- PyTorch
- Keras
- MXNet
- Caffe

These datasets are included in the `data/` directory of the repository.

## Getting Started

### Option 1: Google Colab (Recommended)

1. Upload `ISE.ipynb` to Google Colab
2. Configure the runtime to use GPU
3. Run all cells to set up directories, install dependencies, and run the experiment

### Option 2: Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Alva1103/ISE-Coursework.git
   cd ISE-Coursework
   ```

2. Set up a Python 3.8+ virtual environment and install dependencies:
   ```bash
   python -m venv ise_env
   source ise_env/bin/activate  # On Windows: ise_env\Scripts\activate
   pip install transformers==4.36.2 datasets==2.15.0 scikit-learn==1.2.2 pandas==1.5.3 numpy==1.24.0 matplotlib==3.7.2 seaborn==0.12.2 torch==2.1.2 nltk==3.8.1 tqdm==4.66.1
   ```

3. Run the notebook or convert it to a Python script:
   ```bash
   jupyter notebook ISE.ipynb
   # Or
   jupyter nbconvert --to script ISE.ipynb
   python ISE.py
   ```

## Documentation

- **requirements.pdf**: Details all dependencies and requirements
- **manual.pdf**: Provides usage instructions
- **replication.pdf**: Contains step-by-step instructions to replicate the results

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (strongly recommended)
- **RAM**: At least 16GB
- **Storage**: At least 10GB free space

## Citation

If you use this code or methodology in your research, please cite:

```
@article{ISE2025,
  title={Fine-Tuning RoBERTa for Performance-Related Bug Report Classification},
  author={Joy Alvaro Siahaan},
  year={2025}
}
```

## License

[MIT License](https://opensource.org/licenses/MIT)
