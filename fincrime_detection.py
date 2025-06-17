#!/usr/bin/env python3
"""
Financial Crime Detection using Gradient Boosting Classifier
============================================================

This script implements a minimal Gradient Boosting Classifier for detecting
suspicious/fraudulent financial transactions using synthetic data.

Features:
- transaction_amount: Amount of the transaction
- time_since_last_transaction: Time elapsed since last transaction (hours)
- merchant_category: Categorical merchant type (encoded)
- account_age: Age of the account in days
- is_foreign_transaction: Binary flag for foreign transactions
- device_mismatch: Binary flag for device mismatches

Target: Binary classification (1 = suspicious/fraudulent, 0 = normal)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic financial transaction data that mimics real-world patterns.

    Args:
        n_samples (int): Number of samples to generate

    Returns:
        pd.DataFrame: Synthetic transaction dataset
    """
    print(f"Generating {n_samples} synthetic transaction records...")

    # Define merchant categories
    merchant_categories = [
        'grocery', 'gas_station', 'restaurant', 'online_retail',
        'atm', 'pharmacy', 'hotel', 'airline', 'entertainment', 'other'
    ]

    data = []

    for _ in range(n_samples):
        # Generate features with realistic distributions

        # Transaction amount (log-normal distribution for realistic skew)
        if np.random.random() < 0.05:  # 5% high-value transactions
            transaction_amount = np.random.lognormal(mean=6, sigma=1)  # Higher amounts
        else:
            transaction_amount = np.random.lognormal(mean=3, sigma=1)  # Normal amounts

        # Time since last transaction (exponential distribution)
        time_since_last = np.random.exponential(scale=24)  # Average 24 hours

        # Merchant category
        merchant_category = np.random.choice(merchant_categories)

        # Account age in days (normal distribution, minimum 30 days)
        account_age = max(30, int(np.random.normal(loc=365, scale=200)))

        # Foreign transaction (10% probability)
        is_foreign_transaction = np.random.random() < 0.1

        # Device mismatch (5% probability for normal, higher for suspicious)
        device_mismatch = np.random.random() < 0.05

        # Generate label based on suspicious patterns
        suspicious_score = 0

        # High transaction amounts are more suspicious
        if transaction_amount > 5000:
            suspicious_score += 0.3
        elif transaction_amount > 2000:
            suspicious_score += 0.1

        # Very short time since last transaction can be suspicious
        if time_since_last < 0.5:  # Less than 30 minutes
            suspicious_score += 0.4

        # Foreign transactions are slightly more suspicious
        if is_foreign_transaction:
            suspicious_score += 0.2

        # Device mismatch is highly suspicious
        if device_mismatch:
            suspicious_score += 0.5

        # New accounts are more suspicious for high amounts
        if account_age < 90 and transaction_amount > 1000:
            suspicious_score += 0.3

        # ATM transactions at odd hours with device mismatch
        if merchant_category == 'atm' and device_mismatch:
            suspicious_score += 0.4

        # Online retail with foreign transaction and device mismatch
        if merchant_category == 'online_retail' and is_foreign_transaction and device_mismatch:
            suspicious_score += 0.6

        # Convert suspicious score to binary label with some randomness
        is_fraudulent = (suspicious_score + np.random.normal(0, 0.1)) > 0.5

        data.append({
            'transaction_amount': round(transaction_amount, 2),
            'time_since_last_transaction': round(time_since_last, 2),
            'merchant_category': merchant_category,
            'account_age': account_age,
            'is_foreign_transaction': int(is_foreign_transaction),
            'device_mismatch': int(device_mismatch),
            'is_fraudulent': int(is_fraudulent)
        })

    df = pd.DataFrame(data)
    print(f"Dataset generated with {len(df)} records")
    print(f"Fraud rate: {df['is_fraudulent'].mean():.2%}")

    return df

def preprocess_data(df):
    """
    Preprocess the data for machine learning.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        tuple: X (features), y (target)
    """
    print("Preprocessing data...")

    # Encode categorical variables
    le = LabelEncoder()
    df_processed = df.copy()
    df_processed['merchant_category_encoded'] = le.fit_transform(df['merchant_category'])

    # Select features
    feature_columns = [
        'transaction_amount',
        'time_since_last_transaction',
        'merchant_category_encoded',
        'account_age',
        'is_foreign_transaction',
        'device_mismatch'
    ]

    X = df_processed[feature_columns]
    y = df_processed['is_fraudulent']

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, le

def train_gradient_boosting_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        GradientBoostingClassifier: Trained model
    """
    print("Training Gradient Boosting Classifier...")

    # Initialize the model with optimized parameters for financial fraud detection
    gb_model = GradientBoostingClassifier(
        n_estimators=100,           # Number of boosting stages
        learning_rate=0.1,          # Learning rate
        max_depth=6,                # Maximum depth of trees
        min_samples_split=20,       # Minimum samples to split
        min_samples_leaf=10,        # Minimum samples per leaf
        subsample=0.8,              # Fraction of samples for fitting individual learners
        random_state=42,            # For reproducibility
        verbose=1                   # Print progress
    )

    # Train the model
    gb_model.fit(X_train, y_train)

    print("Model training completed!")
    return gb_model

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the trained model and print comprehensive metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {2 * (precision * recall) / (precision + recall):.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraudulent']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30} {row['importance']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance
    }

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance.

    Args:
        feature_importance: DataFrame with feature importance
        top_n: Number of top features to plot
    """
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features for Fraud Detection')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/Users/anilkumar/Documents/03.Learning/03.05 Python/AML/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the complete financial crime detection pipeline.
    """
    print("="*80)
    print("FINANCIAL CRIME DETECTION USING GRADIENT BOOSTING")
    print("="*80)
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate synthetic data
    df = generate_synthetic_data(n_samples=10000)

    # Display basic statistics
    print("\nDataset Overview:")
    print(df.describe())
    print("\nDataset Info:")
    print(df.info())

    # Preprocess data
    X, y, _ = preprocess_data(df)

    # Split data into training and testing sets
    print("\nSplitting data into train/test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate: {y_test.mean():.2%}")

    # Train the model
    model = train_gradient_boosting_model(X_train, y_train)

    # Evaluate the model
    results = evaluate_model(model, X_test, y_test, X.columns.tolist())

    # Plot feature importance
    plot_feature_importance(results['feature_importance'])

    # Save the dataset for further analysis
    df.to_csv('/Users/anilkumar/Documents/03.Learning/03.05 Python/AML/synthetic_transactions.csv', index=False)
    print("\nDataset saved to: synthetic_transactions.csv")

    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
