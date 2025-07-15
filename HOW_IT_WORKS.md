# How the AML Financial Crime Detection System Works

## Overview

This system implements an **Anti-Money Laundering (AML)** fraud detection pipeline using machine learning. It generates synthetic financial transaction data and trains a **Gradient Boosting Classifier** to identify potentially fraudulent transactions.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Data Generation│ -> │  Feature Engineering│ -> │  Model Training │ -> │   Evaluation     │
│  (Synthetic)    │    │  & Preprocessing    │    │  (Gradient Boost)│    │  & Visualization │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## 📊 Data Generation Process

### 1. **Synthetic Transaction Creation**
The system generates realistic financial transactions with these features:

- **Transaction Amount**: Log-normal distribution (most small, few large transactions)
- **Time Since Last Transaction**: Exponential distribution (average 24 hours)
- **Merchant Categories**: 10 types (grocery, gas_station, restaurant, online_retail, atm, pharmacy, hotel, airline, entertainment, other)
- **Account Age**: Normal distribution around 365 days (minimum 30 days)
- **Foreign Transaction**: 10% probability binary flag
- **Device Mismatch**: 5% probability binary flag

### 2. **Fraud Label Generation (Rule-Based)**
The system creates fraud labels using a **suspicious scoring system**:

| Pattern | Suspicion Score | Logic |
|---------|----------------|--------|
| High amount (>$5000) | +0.3-0.4 | Large transactions are riskier |
| Quick transactions (<30 min) | +0.3-0.4 | Rapid successive transactions suspicious |
| Foreign transactions | +0.2 | Cross-border adds risk |
| Device mismatch | +0.5 | **Strongest indicator** of account compromise |
| New account + high amount | +0.3 | New accounts with large transactions risky |
| ATM + device mismatch | +0.4 | ATM fraud pattern |
| Online retail + foreign + device mismatch | +0.6 | Complex fraud pattern |

**Final fraud determination**: `(suspicious_score + random_noise) > threshold`

## 🔧 Feature Engineering

The system uses **6 key features** for fraud detection:

1. **`transaction_amount`**: Dollar amount of transaction
2. **`time_since_last_transaction`**: Hours since previous transaction
3. **`merchant_category_encoded`**: Categorical merchant type (label encoded 0-9)
4. **`account_age`**: Days since account creation
5. **`is_foreign_transaction`**: Binary flag (0/1)
6. **`device_mismatch`**: Binary flag (0/1)

## 🤖 Machine Learning Model

### **Algorithm: Gradient Boosting Classifier**
- **Type**: Ensemble method combining multiple decision trees
- **Boosting**: Each tree learns from previous trees' mistakes
- **Parameters**:
  - Estimators: 50-100 trees
  - Learning rate: 0.1
  - Max depth: 4-6 levels
  - Subsample: 0.8 (80% of data per tree)

### **Training Process**:
1. **Data Split**: 80% training, 20% testing (stratified)
2. **Model Fitting**: Gradient boosting on training data
3. **Prediction**: Binary classification (0=normal, 1=fraud)
4. **Evaluation**: Multiple metrics calculation

## 📈 Performance Metrics

### **Current Results**:
- **Accuracy**: ~98.6% (overall correct predictions)
- **Precision**: ~84% (of flagged transactions, 84% are actually fraud)
- **Recall**: ~88% (catches 88% of actual fraud cases)
- **F1-Score**: ~86% (harmonic mean of precision/recall)

### **Feature Importance**:
1. **`device_mismatch`**: 72% - Most important fraud indicator
2. **`time_since_last_transaction`**: 10% 
3. **`transaction_amount`**: 9%
4. **`account_age`**: 4%
5. **`is_foreign_transaction`**: 4%
6. **`merchant_category_encoded`**: 1%

## 🎯 Key Insights

### **Strongest Fraud Indicators**:
1. **Device Mismatch** - Account accessed from unusual device
2. **Transaction Timing** - Very quick successive transactions
3. **Transaction Amount** - Unusually large amounts
4. **Account Age** - New accounts with high-value transactions

### **Fraud Patterns Detected**:
- Account takeover (device mismatch)
- Transaction velocity attacks (rapid transactions)
- Money laundering (large unusual amounts)
- Cross-border fraud (foreign + device mismatch)

## 🚀 Potential Improvements

### **Technical Enhancements**:
1. **Real-time Scoring**: API for single transaction evaluation
2. **Model Persistence**: Save/load trained models
3. **Hyperparameter Tuning**: Grid search optimization
4. **Cross-validation**: More robust evaluation
5. **Ensemble Methods**: Combine multiple algorithms

### **Feature Engineering**:
1. **Rolling Statistics**: Average amounts over time windows
2. **Transaction Velocity**: Transactions per hour/day
3. **Geographic Features**: Location-based risk scoring
4. **Behavioral Patterns**: Usual vs unusual merchant categories
5. **Time-based Features**: Hour of day, day of week patterns

### **Data Improvements**:
1. **More Realistic Patterns**: Time-of-day effects, seasonal trends
2. **Class Imbalance Handling**: SMOTE, class weights
3. **Feature Scaling**: Standardization for sensitive algorithms
4. **Data Validation**: Input quality checks

### **Production Readiness**:
1. **Configuration System**: YAML/JSON parameter files
2. **Logging Framework**: Replace print statements
3. **Error Handling**: Graceful failure recovery
4. **Unit Testing**: Comprehensive test coverage
5. **Model Monitoring**: Drift detection and retraining
6. **API Development**: REST endpoints for integration

## 🔍 Code Structure

```
AML/
├── fincrime_detection.py         # Full-featured implementation
├── simple_fincrime_detection.py  # Minimal implementation
├── requirements.txt              # Python dependencies
├── README.md                     # Basic usage guide
├── HOW_IT_WORKS.md              # This detailed explanation
└── [Generated files]
    ├── synthetic_transactions.csv # Generated dataset
    └── feature_importance.png     # Feature importance plot
```

## 🎮 Usage Examples

### **Basic Usage**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run simple version
python simple_fincrime_detection.py

# Run full version with visualizations
python fincrime_detection.py
```

### **Expected Output**:
- Console metrics (accuracy, precision, recall)
- Feature importance rankings
- Classification report
- Generated CSV dataset
- Feature importance visualization

## 💡 Business Value

This system helps financial institutions:
1. **Reduce Fraud Losses**: Early detection of suspicious transactions
2. **Regulatory Compliance**: AML/KYC compliance automation
3. **Customer Protection**: Prevent account takeovers
4. **Operational Efficiency**: Automated risk scoring
5. **Risk Management**: Real-time transaction monitoring

## 🔄 Next Steps for Enhancement

1. **Fix immediate issues** (file paths, configuration)
2. **Add model persistence** for production use
3. **Implement real-time inference** capabilities
4. **Enhance feature engineering** with time-series patterns
5. **Build production API** for integration
6. **Add comprehensive testing** and monitoring
7. **Create deployment pipeline** for model updates

This foundation provides a solid starting point for building a production-ready financial crime detection system!