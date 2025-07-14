# Detecting-AI-Generated-Text-using-Transformer-Models
This repository contains two notebooks that explore AI-generated text detection using a hybrid approach of transformer-based deep learning models and linguistic feature-based machine learning models.

🔍 Project Overview
With the widespread use of generative language models like ChatGPT, Bard, and LLaMA, it's essential to distinguish AI-generated text from human-written content in various contexts such as education, journalism, and content moderation.

This project explores two methodologies:

Transformer-Based Classification (Generative_transforms.ipynb)
Fine-tuning Hugging Face models to classify text as AI or Human.

LLM + Linguistic Feature-Based ML (Meghana_and_Mounika_llm_detect_ai_generated_text.ipynb)
Using models like BERt and linguistic cues such as punctuation usage, POS diversity, readability scores, and syntactic complexity along with traditional ML models like Logistic Regression, XGBoost, and Random Forest.

📄 Reference
This project is inspired by the paper: https://aclanthology.org/2024.icon-1.21/

Yadagiri et al., ICON 2024 — Detecting AI-Generated Text with Pre-Trained Models Using Linguistic Features

📊 Dataset
The dataset is compiled from multiple sources on Kaggle, and contains:

Mixed Dataset (Training dataset with reduced rows)
Testing Dataset (Testing dataset with reduced rows)
Combined raw dataset of ~29,000 rows
submissions.csv (contains the outputs)
All data files are provided in the data/ directory.

🧠 Features Extracted
Readability Metrics: Flesch-Kincaid, sentence complexity
POS Tag Analysis
Punctuation Distribution
Token Counts and Diversity
Transformer Embeddings (via Hugging Face)


📈 Model Results Summary
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression + Features	~87%	High	Moderate	Good
Fine-tuned BERT	~93%	High	High	High
Ensemble (Features + BERT)	~94%	Very High	High	Excellent


🚀 Getting Started
Clone this repository:
git clone https://github.com/your-username/ai-text-detector.git
cd ai-text-detector
Create a virtual environment (optional):
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:
pip install -r requirements.txt
Run the notebooks:
Open Generative_transforms.ipynb for transformer-based classification
Open Meghana_and_Mounika_llm_detect_ai_generated_text.ipynb for feature-based ML classification


🧪 Libraries Used
Hugging Face Transformers
Scikit-learn
Pandas, NumPy, Seaborn, Matplotlib
NLTK, spaCy, textstat
PyTorch
LightGBM & XGBoost


📬 Authors
Meghana Naidu Thokala – VIT-AP University
Mounika Garnedi – VIT-AP University
