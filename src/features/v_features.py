import pandas as pd

def create_v_features(df):
    """
    Create aggregation features from V columns efficiently.
    Uses data-driven blocks discovered via missing-value EDA.
    """
    print("Creating V-column aggregation features...")
    
    v_cols = [col for col in df.columns if col.startswith('V')]
    if not v_cols:
        return df
        
    # The actual groups validated by missing-value patterns
    v_groups = {
        'v1_auth': [f'V{i}' for i in range(1, 12)],      # V1 to V11
        'v2_behav': [f'V{i}' for i in range(12, 35)],    # V12 to V34
        'v3_hist': [f'V{i}' for i in range(35, 53)],     # V35 to V52
        'v4_card': [f'V{i}' for i in range(53, 75)],     # V53 to V74
        'v5_geo': [f'V{i}' for i in range(75, 95)],      # V75 to V94
        'v6_item': [f'V{i}' for i in range(95, 138)],    # V95 to V137
        'v7_dev1': [f'V{i}' for i in range(138, 167)],   # V138 to V166
        'v8_dev2': [f'V{i}' for i in range(167, 217)],   # V167 to V216 
        'v9_net1': [f'V{i}' for i in range(217, 279)],   # V217 to V278 
        'v10_net2': [f'V{i}' for i in range(279, 322)],  # V279 to V321 
        'v11_stat': [f'V{i}' for i in range(322, 340)],  # V322 to V339
    }
    
    new_features = {}
    
    for group_name, group_cols in v_groups.items():
        existing_cols = [col for col in group_cols if col in df.columns]
        
        if existing_cols:
            subset = df[existing_cols]
            new_features[f'{group_name}_sum'] = subset.sum(axis=1)
            new_features[f'{group_name}_mean'] = subset.mean(axis=1)
            new_features[f'{group_name}_std'] = subset.std(axis=1)
            new_features[f'{group_name}_nan_count'] = subset.isna().sum(axis=1)
            
    # Overall V statistics across all 339 columns
    if v_cols:
        subset_all = df[v_cols]
        new_features['V_sum_all'] = subset_all.sum(axis=1)
        new_features['V_mean_all'] = subset_all.mean(axis=1)
        new_features['V_std_all'] = subset_all.std(axis=1)
        new_features['V_nan_count_all'] = subset_all.isna().sum(axis=1)
        new_features['V_nan_ratio'] = new_features['V_nan_count_all'] / len(v_cols)
        
    # Concatenate
    new_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, new_df], axis=1)