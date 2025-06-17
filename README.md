# Financial Crime Detection using Gradient Boosting

A minimal implementation of a Gradient Boosting Classifier for detecting suspicious/fraudulent financial transactions using synthetic data.

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
├── fincrime_detection.py     # Main script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── synthetic_transactions.csv # Generated dataset (after running)
└── feature_importance.png    # Feature importance plot (after running)
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
