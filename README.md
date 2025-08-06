# Lettuce Leaf Classification: A Multi-Modal Machine Learning Approach

## Project Overview

This repository implements a comprehensive machine learning pipeline for automated lettuce leaf disease classification using both traditional machine learning techniques and deep learning approaches. The project demonstrates end-to-end model development, from exploratory data analysis to production-ready deployment with experiment tracking and model versioning.

## Research Objectives

- **Primary Goal**: Develop robust classification models for lettuce leaf disease identification
- **Secondary Goals**: 
  - Compare classical ML vs deep learning performance
  - Implement Vision Transformer (ViT) architecture for image classification
  - Create production-ready inference pipeline with web interface
  - Establish MLOps best practices with experiment tracking

## Technical Architecture

### Dataset Structure
The project utilizes feature-extracted data from lettuce leaf images with the following characteristics:
- **Format**: Pre-extracted features stored in `data/leaf_features.xlsx`
- **Features**: Numerical features derived from image analysis
- **Classes**: Multiple lettuce disease categories
- **Split**: 80% training, 20% validation with stratified sampling

**Note**: Original image dataset should be downloaded from Kaggle (link to be provided by dataset source).

### Methodology Overview

#### 1. Classical Machine Learning Pipeline (`02_Classical_ML_models.ipynb`)
- **Feature Engineering**: StandardScaler normalization
- **Models Implemented**:
  - Support Vector Machine (SVM) with multiple kernels
  - Random Forest Classifier
  - Decision Tree Classifier
  - XGBoost Classifier
  - Ensemble Voting Classifier (top 3 performers)
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Experiment Tracking**: MLflow integration for model versioning

#### 2. Deep Neural Network Approach (`03_DNN_approach.ipynb`)
- **Architecture**: TinyVGG CNN implementation
- **Framework**: PyTorch with modular code structure
- **Training**: Custom training loops with validation tracking
- **Optimization**: Adam optimizer with learning rate scheduling

#### 3. Vision Transformer Implementation (`04_Custom_VIT_approach.ipynb`)
- **Model**: Custom ViT implementation based on "Attention Is All You Need" paper
- **Patch Embedding**: 16x16 pixel patches with learnable position encoding
- **Architecture Components**:
  - Multi-Head Self-Attention (MSA)
  - Multilayer Perceptron (MLP)
  - Transformer Encoder blocks
  - Classification head
- **Input Resolution**: 224x224 pixels as per ViT paper specifications

#### 4. Production Pipeline (`modular/going_modular/`)
- **Modular Design**: Separation of concerns with dedicated modules
- **Components**:
  - `data_setup.py`: DataLoader creation and preprocessing
  - `model_builder.py`: TinyVGG architecture definition
  - `engine.py`: Training and evaluation loops
  - `prediction.py`: Inference utilities
  - `utils.py`: Helper functions
- **API Development**: FastAPI REST API for model serving
- **Web Interface**: Streamlit application for interactive predictions

## Technology Stack

### Core Frameworks
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch, torchvision
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

### MLOps & Deployment
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI
- **Web Interface**: Streamlit
- **Environment Management**: Python virtual environment

### Development Tools
- **IDE**: Jupyter Notebooks for experimentation
- **Version Control**: Git
- **Dependency Management**: pip requirements

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for accelerated training)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Lettuce_leaf_classification
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv leaf_classf

# Activate virtual environment
# Windows:
leaf_classf\Scripts\activate
# Linux/macOS:
source leaf_classf/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install xgboost mlflow fastapi uvicorn streamlit
pip install jupyter notebook ipykernel
pip install pillow opencv-python
pip install tqdm joblib
```

### Step 4: Download Dataset
1. Download the lettuce leaf disease dataset from Kaggle
2. Extract the dataset to appropriate directory structure
3. Ensure `data/leaf_features.xlsx` contains the feature-extracted data

### Step 5: Jupyter Environment Setup
```bash
# Add virtual environment to Jupyter
python -m ipykernel install --user --name=leaf_classf --display-name "Lettuce Classification"

# Launch Jupyter
jupyter notebook
```

## Usage Instructions

### 1. Exploratory Data Analysis
```bash
# Start with data visualization
jupyter notebook 00_data_visualization.ipynb

# Proceed with comprehensive EDA
jupyter notebook 01_EDA.ipynb
```

### 2. Classical Machine Learning Training
```bash
# Train and evaluate classical ML models
jupyter notebook 02_Classical_ML_models.ipynb
```

This notebook will:
- Train multiple ML algorithms with hyperparameter tuning
- Generate confusion matrices and ROC curves
- Log experiments to MLflow
- Save the best performing ensemble model

### 3. Deep Learning Approaches
```bash
# Train TinyVGG CNN model
jupyter notebook 03_DNN_approach.ipynb

# Implement and train Vision Transformer
jupyter notebook 04_Custom_VIT_approach.ipynb
```

### 4. Model Deployment

#### FastAPI REST API
```bash
# Navigate to modular directory
cd modular/going_modular

# Start API server
python main.py
# or
uvicorn main:app --reload
```

#### Streamlit Web Interface
```bash
# Launch Streamlit app
streamlit run pages/comparison.py
```

### 5. Experiment Tracking
```bash
# View MLflow experiments
mlflow ui
# Navigate to http://localhost:5000
```

## Project Structure

```
Lettuce_leaf_classification/
├── data/
│   └── leaf_features.xlsx          # Feature-extracted dataset
├── notebooks/
│   ├── 00_data_visualization.ipynb # Initial data exploration
│   ├── 01_EDA.ipynb               # Exploratory data analysis
│   ├── 02_Classical_ML_models.ipynb # Traditional ML approaches
│   ├── 03_DNN_approach.ipynb      # Deep neural networks
│   └── 04_Custom_VIT_approach.ipynb # Vision Transformer
├── modular/
│   └── going_modular/
│       ├── data_setup.py          # Data loading utilities
│       ├── model_builder.py       # Model architectures
│       ├── engine.py              # Training/evaluation
│       ├── prediction.py          # Inference pipeline
│       ├── main.py                # FastAPI application
│       └── utils.py               # Helper functions
├── pages/
│   └── comparison.py              # Streamlit web interface
├── mlruns/                        # MLflow experiment logs
├── models/                        # Saved model weights
├── helper_functions.py            # Utility functions
├── installation_guide.txt         # Setup instructions
└── README.md                      # Project documentation
```

## Model Performance

### Classical Machine Learning Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SVM (RBF) | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Random Forest | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| XGBoost | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| Voting Ensemble | 0.96 | 0.93 | 0.96 | 0.94 | 0.99 |

### Deep Learning Results
- **TinyVGG CNN**: Achieved competitive performance with efficient architecture
- **Vision Transformer**: State-of-the-art attention-based classification

## API Documentation

### FastAPI Endpoints

#### POST /predict
Classify lettuce leaf disease from uploaded image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/lettuce_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "healthy",
  "confidence": 0.95,
  "probability": 0.95
}
```

## Reproducibility

### Experiment Reproduction
1. **Environment**: Use provided virtual environment setup
2. **Random Seeds**: Set in all relevant components for deterministic results
3. **Data Splits**: Stratified sampling with fixed random state
4. **Model Training**: MLflow tracking ensures reproducible experiments

### Model Artifacts
- **Classical ML**: Saved using joblib (`Lctf_voting_model.pkl`)
- **Deep Learning**: PyTorch model weights (`TinyVGG_DNN_model_0_weights.pth`)
- **Preprocessing**: StandardScaler saved as `scaler.pkl`

## Research Contributions

1. **Comprehensive Comparison**: Classical ML vs Deep Learning performance analysis
2. **Vision Transformer Implementation**: Custom ViT architecture for agricultural imaging
3. **Production Pipeline**: End-to-end deployment with REST API and web interface
4. **MLOps Integration**: Experiment tracking and model versioning best practices

## Future Enhancements

1. **Data Augmentation**: Implement advanced augmentation techniques
2. **Transfer Learning**: Pre-trained model fine-tuning
3. **Model Optimization**: Quantization and pruning for edge deployment
4. **Real-time Processing**: Stream processing capabilities
5. **Multi-modal Fusion**: Combine image and sensor data

## Technical Specifications

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, NVIDIA GPU with 6GB+ VRAM
- **Storage**: 5GB free space for models and data

### Software Dependencies
- **Python**: 3.8-3.11
- **PyTorch**: >=1.12.0
- **scikit-learn**: >=1.1.0
- **MLflow**: >=2.0.0
- **FastAPI**: >=0.68.0
- **Streamlit**: >=1.12.0

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Vision Transformer implementation based on "An Image is Worth 16x16 Words" paper
- TinyVGG architecture inspired by CNN Explainer
- MLflow integration for experiment tracking
- Kaggle community for dataset provision

## Contact

For technical inquiries or collaboration opportunities, please open an issue in this repository.

---

**Note**: This is a research project demonstrating machine learning methodologies for agricultural applications. Results should be validated in real-world scenarios before production deployment. 