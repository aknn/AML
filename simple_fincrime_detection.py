#!/usr/bin/env python3
"""
Minimal Gradient Boosting Classifier for Financial Crime Detection
================================================================

This script creates synthetic financial transaction data and trains a 
Gradient Boosting Classifier to detect fraudulent transactions.

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic financial transaction data."""
    print(f"Generating {n_samples} synthetic transaction records...")
    
    # Define merchant categories
    merchant_categories = ['grocery', 'gas_station', 'restaurant', 'online_retail', 
                          'atm', 'pharmacy', 'hotel', 'airline', 'entertainment', 'other']
    
    data = []
    
    for _ in range(n_samples):
        # Transaction amount (log-normal distribution)
        if np.random.random() < 0.1:  # 10% high-value transactions
            transaction_amount = np.random.lognormal(mean=6, sigma=1)
        else:
            transaction_amount = np.random.lognormal(mean=3, sigma=1)
        
        # Time since last transaction (exponential distribution)
        time_since_last = np.random.exponential(scale=24)
        
        # Merchant category
        merchant_category = np.random.choice(merchant_categories)
        
        # Account age (normal distribution, minimum 30 days)
        account_age = max(30, int(np.random.normal(loc=365, scale=200)))
        
        # Foreign transaction (10% probability)
        is_foreign_transaction = np.random.random() < 0.1
        
        # Device mismatch (5% probability)
        device_mismatch = np.random.random() < 0.05
        
        # Generate fraud label based on suspicious patterns
        suspicious_score = 0
        
        # High amounts are suspicious
        if transaction_amount > 5000:
            suspicious_score += 0.4
        elif transaction_amount > 2000:
            suspicious_score += 0.2
            
        # Very quick transactions are suspicious
        if time_since_last < 0.5:
            suspicious_score += 0.3
            
        # Foreign transactions are slightly suspicious
        if is_foreign_transaction:
            suspicious_score += 0.2
            
        # Device mismatch is highly suspicious
        if device_mismatch:
            suspicious_score += 0.5
            
        # New accounts with high amounts
        if account_age < 90 and transaction_amount > 1000:
            suspicious_score += 0.3
        
        # Convert to binary with some randomness
        is_fraudulent = (suspicious_score + np.random.normal(0, 0.1)) > 0.4
        
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
    print(f"Dataset generated: {len(df)} records, Fraud rate: {df['is_fraudulent'].mean():.2%}")
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning."""
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
    print(f"Fraud cases: {y.sum()}, Normal cases: {(y==0).sum()}")
    
    return X, y

def main():
    """Main function to run the fraud detection pipeline."""
    print("="*60)
    print("FINANCIAL CRIME DETECTION - GRADIENT BOOSTING")
    print("="*60)
    
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=5000)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train/test split
    print("Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Gradient Boosting Classifier
    print("Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions
    y_pred = gb_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1-Score:  {f1_score:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraudulent']))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = X.columns.tolist()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:30} {row['importance']:.4f}")
    
    # Save dataset
    df.to_csv('/Users/anilkumar/Documents/03.Learning/03.05 Python/AML/synthetic_transactions.csv', index=False)
    print("\nDataset saved to: synthetic_transactions.csv")
    
    print("="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
