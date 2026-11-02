import pandas as pd
import numpy as np

def create_address_features(df):
    """Clean, vectorized address and distance feature extraction"""
    print("Creating address features...")
    
    # 1. Address combinations (Proxy IDs for geographic + product patterns)
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['addr1_addr2'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
    
    if 'addr1' in df.columns and 'ProductCD' in df.columns:
        df['addr1_ProductCD'] = df['addr1'].astype(str) + '_' + df['ProductCD'].astype(str)
    
    # 2. Missing value flags (Keeping as native booleans True/False)
    if 'addr1' in df.columns:
        df['addr1_missing'] = df['addr1'].isna()
    
    if 'addr2' in df.columns:
        df['addr2_missing'] = df['addr2'].isna()
    
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['both_addr_missing'] = df['addr1'].isna() & df['addr2'].isna()
    
    # 3. Distance features (Missing flags + Log Transforms to squash outliers)
    for dist_col in ['dist1', 'dist2']:
        if dist_col in df.columns:
            df[f'{dist_col}_missing'] = df[dist_col].isna()
            
            # log1p safely handles 0s, but we need to fill NAs first to avoid errors
            df[f'{dist_col}_log'] = np.log1p(df[dist_col].fillna(0))
    
    return df