import pandas as pd

def create_aggregation_features(train_df, test_df=None):
    """
    Create frequency and aggregation features safely.
    Prevents Data Leakage by calculating maps STRICTLY from the training set.
    """
    print("Creating aggregation features...")
    
    # 1. Frequency Encoding (How rare is this category?)
    freq_cols = ['card1', 'card2', 'card4', 'card6', 'addr1', 'addr2', 
                 'P_emaildomain', 'R_emaildomain', 'ProductCD']
    
    for col in freq_cols:
        if col in train_df.columns:
            # Learn frequencies from train only
            freq_map = train_df[col].value_counts().to_dict()
            
            # Apply to train
            train_df[f'{col}_freq'] = train_df[col].map(freq_map)
            
            # Apply to test (Fill unseen categories with 0)
            if test_df is not None and col in test_df.columns:
                test_df[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0)
    
    # 2. Transaction Amount Aggregations (Is this normal for this user?)
    agg_cols = ['card1', 'card2', 'addr1']
    
    for col in agg_cols:
        if col in train_df.columns:
            # Learn historical behavior from train only
            mean_map = train_df.groupby(col)['TransactionAmt'].mean().to_dict()
            std_map = train_df.groupby(col)['TransactionAmt'].std().to_dict()
            
            # Apply to train
            train_df[f'{col}_TransactionAmt_mean'] = train_df[col].map(mean_map)
            train_df[f'{col}_TransactionAmt_std'] = train_df[col].map(std_map)
            train_df[f'{col}_TransactionAmt_dev'] = train_df['TransactionAmt'] - train_df[f'{col}_TransactionAmt_mean']
            
            # Apply to test (Fill unseen cards with the global average)
            if test_df is not None and col in test_df.columns:
                global_mean = train_df['TransactionAmt'].mean()
                test_df[f'{col}_TransactionAmt_mean'] = test_df[col].map(mean_map).fillna(global_mean)
                test_df[f'{col}_TransactionAmt_std'] = test_df[col].map(std_map).fillna(0)
                test_df[f'{col}_TransactionAmt_dev'] = test_df['TransactionAmt'] - test_df[f'{col}_TransactionAmt_mean']
                
    return train_df, test_df