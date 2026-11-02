import pandas as pd
import numpy as np

def create_c_d_features(df):
    """Clean, fragmentation-safe feature extraction for Counts and Deltas"""
    print("Creating C and D column features...")
    
    new_features = {}
    
    # C Columns

    c_cols = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
    
    if c_cols:
        subset_c = df[c_cols]
        new_features['C_sum'] = subset_c.sum(axis=1)
        new_features['C_mean'] = subset_c.mean(axis=1)
        new_features['C_std'] = subset_c.std(axis=1)
        new_features['C_max'] = subset_c.max(axis=1)
        new_features['C_min'] = subset_c.min(axis=1)
        
        # Log transform every C column to handle bot-driven outliers
        for col in c_cols:
            new_features[f'{col}_log'] = np.log1p(subset_c[col].fillna(0))

            
    # D Columns (Time Deltas)

    d_cols = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()]
    
    if d_cols:
        subset_d = df[d_cols]
        new_features['D_sum'] = subset_d.sum(axis=1)
        new_features['D_mean'] = subset_d.mean(axis=1)
        new_features['D_std'] = subset_d.std(axis=1)
        new_features['D_nan_count'] = subset_d.isna().sum(axis=1)
        
        # Extract specific missing flags and logs for known critical D columns
        for col in ['D1', 'D15']:
            if col in subset_d.columns:
                new_features[f'{col}_missing'] = subset_d[col].isna()
                new_features[f'{col}_log'] = np.log1p(subset_d[col].fillna(0))
                
    # Concatenate all new features
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)
        
    return df