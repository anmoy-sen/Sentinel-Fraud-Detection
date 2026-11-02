import pandas as pd
import numpy as np

def create_transaction_amount_features(df):
    """Create features based on transaction amount"""
    print("Creating transaction amount features...")
    
    # 1. Log transform to handle extreme values
    df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])
    
    # 2. Extract just the cents (e.g., 45.99 -> 99)
    # Multiplying by 100, rounding to fix precision issues, and using modulo 100
    cents = (df['TransactionAmt'] * 100).round() % 100
    df['TransactionAmt_cents'] = cents
    
    # 3. Create True/False flags for common patterns 
    # (Models like XGBoost handle booleans perfectly, no need to force them to 1/0 integers)
    df['TransactionAmt_is_round'] = (cents == 0)
    df['TransactionAmt_ends_99'] = (cents == 99)
    df['TransactionAmt_ends_95'] = (cents == 95)
    
    # 4. Bin the amounts into discrete categories
    # labels=False automatically assigns integers (0, 1, 2...) to the bins
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000, np.inf]
    df['TransactionAmt_bin'] = pd.cut(df['TransactionAmt'], bins=bins, labels=False)
    
    return df