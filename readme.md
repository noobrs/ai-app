# Movie Review Sentiment Analysis With Naive Bayes, ANN, and DistilBERT

A comprehensive sentiment analysis application that compares three different machine learning approaches: **Naive Bayes**, **Artificial Neural Networks (ANN)**, and **DistilBERT** transformer models. The project includes both model training notebooks and a user-friendly Streamlit web application for real-time sentiment prediction.

## 📋 Table of Contents

- [Features](#features)
- [Models Overview](#models-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## ✨ Features

### 🚀 **Multi-Model Comparison**
- **Naive Bayes**: Traditional probabilistic classifier with extensive NLP preprocessing
- **ANN**: Deep neural network with TF-IDF features
- **DistilBERT**: State-of-the-art transformer model with sliding window technique

### 📊 **Interactive Web Application**
- **Smart Input Detection**: Single review, multiple reviews, or CSV file upload
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Dynamic Visualizations**: 
  - Pie charts and bar charts for sentiment distribution
  - Stacked bar charts for confidence breakdown
  - Grouped comparison charts across models
- **Export Functionality**: Download results as CSV files

### 🎯 **Flexible Analysis Modes**
- **Single Text + Single Model**: Clean result display with confidence metrics
- **Single Text + All Models**: Side-by-side model comparison with stacked confidence chart
- **Multiple Texts + Single Model**: Distribution charts and detailed results grid
- **Multiple Texts + All Models**: Individual model analysis plus comprehensive comparison

## 🧠 Models Overview

### 1. **Naive Bayes Model**
- **Preprocessing**: Extensive NLP pipeline (lemmatization, stopword removal, contraction expansion)
- **Features**: TF-IDF with unigrams and bigrams
- **Strengths**: Fast, interpretable, handles negations well
- **Performance**: 88.53% accuracy

### 2. **Artificial Neural Network (ANN)**
- **Architecture**: Multi-layer perceptron with TF-IDF features
- **Features**: 30,000 feature vocabulary with n-grams
- **Strengths**: Non-linear pattern recognition, memory efficient
- **Performance**: 90.77% accuracy

### 3. **DistilBERT (with Sliding Window)**
- **Architecture**: Transformer-based model with attention mechanism
- **Innovation**: Sliding window technique for long text processing
- **Features**: Contextual embeddings, subword tokenization
- **Performance**: 93.93% accuracy (with sliding window)

## 📁 Project Structure

```
ai-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── comparison.ipynb           # Model comparison visualizations
├── dataset/
│   ├── IMDB Dataset.csv       # Original IMDB movie reviews dataset
│   └── imdb_clean_split.csv   # Preprocessed and split dataset
├── models/
│   ├── nb/                    # Naive Bayes model files
│   ├── ann/                   # ANN model files
│   └── distilbert/            # DistilBERT model variants
├── naivebayes.ipynb           # Naive Bayes training notebook
├── testANN_Real.ipynb         # ANN training notebook
└── distilbert.ipynb           # DistilBERT training notebook
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/noobrs/ai-app.git
cd ai-app
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Check if Streamlit is installed correctly
streamlit --version

# Verify Python dependencies
python -c "import streamlit, transformers, tensorflow, sklearn; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Running the Web Application

1. **Start the Streamlit application:**
```bash
streamlit run app.py
```

2. **Open your web browser** and navigate to:
```
http://localhost:8501
```

### Using the Application

#### **Method 1: Single Review Analysis**
1. Paste your movie review in the text area
2. Select your preferred model or "All Models" for comparison
3. Click "Run" to get instant sentiment analysis
4. View results with confidence scores and visualizations

#### **Method 2: Multiple Reviews Analysis**
1. Enter multiple reviews (one per line) in the text area
2. Select your analysis mode
3. Click "Run" to analyze all reviews
4. Explore distribution charts and detailed results

#### **Method 3: CSV File Upload**
1. Click "Upload CSV file" and select your file
2. Choose the column containing the review text
3. Select your model preference
4. Click "Run" for batch analysis
5. Download results as CSV file

### Example Usage

#### Single Review Example:
```
Input: "This movie was absolutely fantastic! The acting was superb and the plot was engaging."

Output:
✅ POSITIVE
Confidence: 0.9234
P(positive) = 0.9234
```

#### Multiple Reviews Example:
```
Input:
This movie was terrible. Waste of time.
Amazing film! Loved every minute of it.
Not bad, could be better though.

Output:
- Sentiment distribution charts
- Individual predictions for each review
- Downloadable CSV results
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 88.53% | 88.54% | 88.53% | 88.53% |
| **ANN** | 90.77% | 90.78% | 90.77% | 90.77% |
| **DistilBERT** | 93.79% | 93.79% | 93.79% | 93.79% |
| **DistilBERT + Sliding Window** | **93.94%** | **93.96%** | **93.94%** | **93.94%** |

### Key Insights:
- **DistilBERT with Sliding Window** achieves the highest performance
- **Sliding window technique** improves DistilBERT by 0.14% accuracy
- **All models** show consistent performance across different metrics
- **Progressive improvement** from traditional ML to deep learning approaches

## 🔧 Advanced Configuration

### Model Parameters

#### Naive Bayes Configuration:
```python
# Text preprocessing
- Lemmatization with NLTK
- Contraction expansion
- Smart stopword removal (preserves negators)
- TF-IDF: unigrams + bigrams, min_df=2, max_df=0.95
```

#### ANN Configuration:
```python
# Architecture
- Input: TF-IDF features (30,000 dimensions)
- Hidden layers: Dense layers with dropout
- Output: Binary classification
- Optimizer: Adam with learning rate scheduling
```

#### DistilBERT Configuration:
```python
# Standard mode
- Max length: 512 tokens
- Truncation: True
- Padding: max_length

# Sliding window mode
- Max length: 512 tokens
- Stride: 256 tokens
- Max windows: 4 per document
```

### Environment Variables

You can customize the application behavior by setting these environment variables:

```bash
# Model paths (if using local models)
export NB_PATH="path/to/naive_bayes_model"
export ANN_PATH="path/to/ann_model" 
export DISTILBERT_PATH="path/to/distilbert_model"

# DistilBERT parameters
export MAX_LEN=512
export STRIDE=256
```

## 🙏 Acknowledgments

- **IMDB Dataset**: Large Movie Review Dataset by Stanford AI Lab
- **Hugging Face**: Pre-trained DistilBERT model and transformers library
- **Streamlit**: Framework for the web application
- **scikit-learn**: Traditional machine learning algorithms
- **TensorFlow**: Deep learning framework for ANN implementation

---

**Happy Sentiment Analysis! 🎬✨**

*Built with ❤️ using Python, Streamlit, and state-of-the-art NLP models*