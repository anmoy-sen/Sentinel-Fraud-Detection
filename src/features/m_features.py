import pandas as pd

def create_m_features(df):
    """Clean, fragmentation-safe feature extraction for M (Match) columns"""
    print("Creating M column features...")
    
    m_cols = [f'M{i}' for i in range(1, 10)]
    existing_m_cols = [col for col in m_cols if col in df.columns]
    
    if not existing_m_cols:
        return df
        
    new_features = {}
    
    # We will store the boolean arrays temporarily to sum them up later
    # without having to re-query the DataFrame
    t_masks = []
    f_masks = []
    missing_masks = []
    
    for col in existing_m_cols:
        # Creating boolean masks is Memory efficient)
        is_t = (df[col] == 'T')
        is_f = (df[col] == 'F')
        is_missing = df[col].isna()
        
        # Save to new features dict
        new_features[f'{col}_isT'] = is_t
        new_features[f'{col}_isF'] = is_f
        new_features[f'{col}_missing'] = is_missing
        
        # Keep track for the aggregate counts
        t_masks.append(is_t)
        f_masks.append(is_f)
        missing_masks.append(is_missing)
        
    # Calculate aggregates by summing the boolean arrays directly
    # (In Python, True sums as 1, False as 0)
    new_features['M_true_count'] = sum(t_masks)
    new_features['M_false_count'] = sum(f_masks)
    new_features['M_missing_count'] = sum(missing_masks)
    
    # Calculate ratio (Adding 0.001 prevents DivisionByZero errors)
    new_features['M_true_ratio'] = new_features['M_true_count'] / (new_features['M_true_count'] + new_features['M_false_count'] + 0.001)
    
    # Concatenate all at once
    new_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, new_df], axis=1)