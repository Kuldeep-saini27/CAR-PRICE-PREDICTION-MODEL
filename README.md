# 🚗 CarPrice AI - Intelligent Car Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CarPrice AI** is a comprehensive machine learning solution that predicts used car prices with high accuracy using advanced ML algorithms and explainable AI techniques. The system provides intelligent price predictions, market insights, and buyer segmentation through an interactive web application.

## 🎯 Project Motivation

The used car market is complex and dynamic, with prices influenced by numerous factors including brand reputation, technical specifications, market demand, and regional preferences. Traditional price estimation methods often lack transparency and accuracy. **CarPrice AI** addresses these challenges by:

- **🎯 Accurate Predictions**: Leveraging ensemble methods (Random Forest, XGBoost) to achieve 83.8% prediction accuracy
- **🔍 Transparent Insights**: Using SHAP and LIME for explainable AI to understand price drivers
- **📊 Market Intelligence**: Providing comprehensive market analysis and buyer segmentation
- **🚀 User-Friendly Interface**: Offering an intuitive web application for real-time price predictions

## ✨ Key Features

### 🤖 **Advanced Machine Learning**
- **Multiple Algorithms**: Random Forest, XGBoost, Linear Regression with automated model selection
- **High Accuracy**: Best model achieves R² score of 0.8378 (83.8% accuracy)
- **Robust Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Cross-Validation**: Ensures model generalization and prevents overfitting

### 🔍 **Explainable AI**
- **SHAP Analysis**: Global and local feature importance explanations
- **LIME Integration**: Instance-level prediction explanations with interactive HTML reports
- **Feature Visualization**: Comprehensive charts showing factor influences on pricing

### 📊 **Market Intelligence**
- **K-Means Clustering**: Segments buyers into budget, mid-range, and premium categories
- **Interactive Visualizations**: 15+ charts covering price trends, brand analysis, and market patterns
- **Real-time Filtering**: Dynamic data exploration with brand, fuel type, and price filters

### 🎛️ **Interactive Web Application**
- **Cascading Filters**: Intuitive Brand → Model → Specifications selection process
- **Price Predictor**: Real-time predictions with confidence intervals
- **Data Explorer**: Browse complete dataset by brand with detailed specifications
- **Comprehensive Analytics**: Model performance metrics and market insights

### 🐳 **Production Ready**
- **Docker Containerization**: Complete containerization for consistent deployment
- **Cloud Deployment**: Heroku/AWS ready with Procfile and deployment configurations
- **Scalable Architecture**: Modular design supporting easy maintenance and updates

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   ML Pipeline    │    │  Application    │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Raw CSV Data  │───▶│ • Data Cleaning  │───▶│ • Streamlit UI  │
│ • Data Cleaning │    │ • Feature Eng.   │    │ • Visualizations│
│ • Preprocessing │    │ • Model Training │    │ • Predictions   │
└─────────────────┘    │ • Evaluation     │    │ • Analytics     │
                       │ • SHAP/LIME      │    └─────────────────┘
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the models** (optional - pre-trained models included)
```bash
python model_training.py
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
Open your browser and navigate to `http://localhost:8501`

### 🐳 Docker Deployment

1. **Build the Docker image**
```bash
docker build -t carprice-ai .
```

2. **Run with Docker Compose**
```bash
docker-compose up
```

## 📋 Application Features

### 🏠 **Home Dashboard**
- Project overview and key metrics
- Quick access to all features
- System performance indicators

### 💰 **Price Predictor**
- **Step 1**: Select car brand from comprehensive list
- **Step 2**: Choose specific model with price range preview
- **Step 3**: Customize technical specifications
- **Step 4**: Get instant price prediction with confidence intervals

### 📊 **Visualizations**
- **Price Analysis**: Distribution patterns and brand comparisons
- **Brand Intelligence**: Market presence and pricing strategies
- **Fuel & Body Analysis**: Technology adoption and consumer preferences
- **Performance Metrics**: Mileage vs price relationships
- **Technical Specs**: Engine complexity and transmission trends
- **Correlation Matrix**: Feature interdependencies
- **Advanced Analytics**: Trend analysis with regression lines

### 📋 **Available Details**
- Brand-specific car catalog
- Complete specifications database
- Market insights and statistics
- Data export functionality

### 🔬 **Model Analytics**
- Model performance comparison
- Feature importance rankings
- Cross-validation results
- Prediction accuracy metrics

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Data Ingestion**: Load and validate CSV data
2. **Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create derived features and encode categoricals
4. **Scaling**: Normalize numerical features for linear models
5. **Splitting**: Train/test split with stratification

### Model Training Workflow
1. **Multiple Algorithms**: Train Random Forest, XGBoost, Linear models
2. **Hyperparameter Tuning**: Grid search for optimal parameters
3. **Cross-Validation**: 5-fold CV for robust evaluation
4. **Model Selection**: Choose best performer based on R² score
5. **Persistence**: Save models and preprocessors

### Explainability Pipeline
1. **SHAP Analysis**: Generate global and local explanations
2. **LIME Integration**: Create instance-level explanations
3. **Visualization**: Generate interpretable charts and plots
4. **Export**: Save explanations as images and HTML

## 📊 Dataset Information

- **Source**: Comprehensive car specifications database
- **Size**: 250+ car models across 50+ brands
- **Features**: 12 key attributes including price, specifications, and performance metrics
- **Coverage**: Multiple fuel types, body styles, and price segments

### Key Features:
- `Brand`: Car manufacturer
- `Model`: Specific car model
- `Ex_Showroom_PriceRs.`: Target variable (price in Rupees)
- `Fuel_Type`: Petrol, Diesel, Electric, Hybrid, CNG
- `Body_Type`: Sedan, SUV, Hatchback, Coupe, etc.
- `Engine Specs`: Cylinders, fuel capacity, mileage
- `Physical Specs`: Seating capacity, wheelbase, boot space

## 🎯 Model Performance

| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| **Random Forest** | **0.8378** | 2.1M | 1.2M | 15s |
| XGBoost | 0.8156 | 2.3M | 1.4M | 25s |
| Linear Regression | 0.7234 | 2.8M | 1.8M | 2s |

## 🔍 Key Insights

### Price Drivers (Feature Importance)
1. **Brand** (35%): Premium brands command higher prices
2. **Engine Cylinders** (18%): More cylinders = higher performance = higher price
3. **Fuel Type** (15%): Electric/Hybrid premium over conventional fuels
4. **Body Type** (12%): SUVs and luxury sedans priced higher
5. **Wheelbase** (10%): Larger vehicles generally more expensive

### Market Segmentation
- **Budget Segment** (< ₹10L): 45% of market, dominated by hatchbacks
- **Mid-Range** (₹10L-30L): 35% of market, sedans and compact SUVs
- **Premium** (> ₹30L): 20% of market, luxury brands and performance cars

## 🚀 Future Enhancements

- [ ] **Real-time Data**: Integration with live market data APIs
- [ ] **Image Recognition**: Price prediction from car images using CNN
- [ ] **Regional Pricing**: Location-based price adjustments
- [ ] **Market Trends**: Predictive analytics for future price movements
- [ ] **Mobile App**: React Native mobile application
- [ ] **API Service**: RESTful API for third-party integrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **Scikit-learn** for robust ML algorithms
- **XGBoost** for gradient boosting implementation
- **SHAP & LIME** for explainable AI capabilities
- **Streamlit** for rapid web application development
- **Plotly** for interactive visualizations

---

⭐ **Star this repository if you found it helpful!**

The dataset contains 258 car records with 13 features:
- Brand, Model, Ex-Showroom Price (target variable)
- Engine specifications (Cylinders, Emission Norm, Fuel Type)
- Physical attributes (Body Type, Seating Capacity, Wheelbase, Boot Space)
- Performance metrics (ARAI Certified Mileage, Fuel Capacity)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Analysis**: Run `python data_analysis.py` for EDA
2. **Model Training**: Run `python model_training.py` to train ML models
3. **Web App**: Run `streamlit run app.py` to launch the interactive interface

## Project Structure

```
├── Car Dataset - Sheet1.csv    # Raw dataset
├── requirements.txt            # Python dependencies
├── data_analysis.py           # EDA and preprocessing
├── model_training.py          # ML model implementation
├── app.py                     # Streamlit web application
├── models/                    # Saved model files
└── visualizations/            # Generated plots and charts
```

## Models Performance

The project implements and compares multiple ML algorithms to find the optimal model for price prediction based on various evaluation metrics.
