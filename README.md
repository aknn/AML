# Financial Crime Detection using Gradient Boosting

A minimal implementation of a Gradient Boosting Classifier for detecting suspicious/fraudulent financial transactions using synthetic data.

## 📖 How It Works

For a **detailed explanation** of how the system works, see **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**

### Quick Overview
This system:
1. **Generates synthetic financial transaction data** with realistic patterns
2. **Creates fraud labels** using rule-based suspicious scoring 
3. **Trains a Gradient Boosting Classifier** on 6 key features
4. **Achieves ~98.6% accuracy** with ~84% precision in fraud detection
5. **Identifies device mismatch as the strongest fraud indicator** (72% feature importance)

## Features

The model uses the following features to detect financial crimes:

1. **transaction_amount**: Amount of the transaction
2. **time_since_last_transaction**: Time elapsed since last transaction (in hours)
3. **merchant_category**: Type of merchant (grocery, gas_station, restaurant, etc.)
4. **account_age**: Age of the account in days
5. **is_foreign_transaction**: Binary flag indicating if transaction is foreign
6. **device_mismatch**: Binary flag indicating if transaction device doesn't match usual device

## Target Variable

- **is_fraudulent**: Binary classification (1 = suspicious/fraudulent, 0 = normal)

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

Run the main script:

```bash
python fincrime_detection.py
```

The script will:
1. Generate 10,000 synthetic transaction records
2. Preprocess the data and encode categorical variables
3. Split data into training (80%) and testing (20%) sets
4. Train a Gradient Boosting Classifier
5. Evaluate the model and print metrics (accuracy, precision, recall)
6. Display feature importance analysis
7. Save results and visualizations

## Output

The script generates:
- **Console output**: Detailed metrics including accuracy, precision, recall, and classification report
- **feature_importance.png**: Visualization of the most important features
- **synthetic_transactions.csv**: The generated synthetic dataset

## Model Performance

The Gradient Boosting Classifier is configured with:
- 100 estimators (boosting stages)
- Learning rate of 0.1
- Maximum depth of 6
- Subsampling of 0.8 for better generalization

Expected performance on synthetic data:
- Accuracy: ~85-95%
- Precision: ~80-90%
- Recall: ~75-85%

## Synthetic Data Generation

The synthetic data mimics realistic financial transaction patterns:
- Transaction amounts follow a log-normal distribution
- Time between transactions follows an exponential distribution
- Fraud labels are generated based on suspicious patterns:
  - High transaction amounts
  - Rapid consecutive transactions
  - Foreign transactions with device mismatches
  - New accounts with high-value transactions
  - ATM transactions with device mismatches

## Project Structure

```
AML/
├── fincrime_detection.py          # Full-featured implementation with visualizations
├── simple_fincrime_detection.py   # Minimal implementation
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration parameters (customizable)
├── README.md                      # This file
├── HOW_IT_WORKS.md               # Detailed explanation of the system
├── synthetic_transactions.csv     # Generated dataset (after running)
└── feature_importance.png         # Feature importance plot (after running)
```

## Key Insights

The model typically identifies these as the most important features for fraud detection:
1. Device mismatch (strongest predictor)
2. Transaction amount
3. Time since last transaction
4. Foreign transaction flag
5. Account age
6. Merchant category

## Future Enhancements

Potential improvements for production use:
- Real-time fraud detection pipeline
- Advanced feature engineering (transaction velocity, spending patterns)
- Ensemble methods combining multiple algorithms
- Integration with external fraud databases
- A/B testing framework for model updates
- Explainable AI for regulatory compliance

### 🚀 Improvement Ideas
For detailed suggestions on enhancing this system, see the **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** file which includes:
- Technical enhancements (real-time scoring, model persistence, hyperparameter tuning)
- Feature engineering improvements (rolling statistics, behavioral patterns)
- Data improvements (realistic patterns, class imbalance handling)
- Production readiness (APIs, monitoring, testing frameworks)

### 📊 Key Insights from Current Implementation
- **Device mismatch** is the strongest fraud indicator (72% feature importance)
- **Transaction timing** and **amount** are secondary indicators
- The system achieves excellent performance on synthetic data but would need real-world validation
- Current fraud detection rules could be enhanced with machine learning-based pattern recognition
