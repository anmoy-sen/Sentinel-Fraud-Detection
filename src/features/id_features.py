import pandas as pd
import numpy as np

def create_id_features(df):
    """Clean, fragmentation-safe extraction for Identity columns"""
    print("Creating ID features...")
    
    new_features = {}

    # Numerical IDs (id_01 to id_11)
    id_num_cols = [f'id_0{i}' for i in range(1, 10)] + ['id_10', 'id_11']
    existing_id_num = [col for col in id_num_cols if col in df.columns]
    
    if existing_id_num:
        subset_num = df[existing_id_num]
        new_features['id_num_nan_count'] = subset_num.isna().sum(axis=1)
        new_features['id_num_mean'] = subset_num.mean(axis=1)
        new_features['id_num_std'] = subset_num.std(axis=1)
        

    # Categorical Statuses (New vs Found)
    status_cols = ['id_12', 'id_15', 'id_16', 'id_28', 'id_29']
    
    for col in status_cols:
        if col in df.columns:
            new_features[f'{col}_isFound'] = (df[col] == 'Found')
            # Only some columns track 'New', but checking it dynamically is safe
            if 'New' in df[col].unique():
                new_features[f'{col}_isNew'] = (df[col] == 'New')


    # String Parsing (id_34)
    if 'id_34' in df.columns:
        # Format is usually 'match_status:1' or 'match_status:2'
        split_vals = df['id_34'].str.split(':', expand=True)
        if 1 in split_vals.columns:
            new_features['id_34_match'] = pd.to_numeric(split_vals[1], errors='coerce').fillna(-1)
        else:
            new_features['id_34_match'] = -1


    # Boolean Flags (id_36, 37, 38)
    for col in ['id_36', 'id_37', 'id_38']:
        if col in df.columns:
            new_features[f'{col}_isT'] = (df[col] == 'T')
            
    # Concatenate
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)
        
    return df