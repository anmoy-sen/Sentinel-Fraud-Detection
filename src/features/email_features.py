import pandas as pd


def create_email_features(df):
    """Clean, vectorized email feature extraction"""
    print("Creating email features...")
    
    common_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'icloud', 'live', 'msn']
    
    # Process Purchaser (P) and Recipient (R) in a loop to avoid copy/pasting code
    for prefix in ['P', 'R']:
        col = f'{prefix}_emaildomain'
        
        if col in df.columns:
            # 1. Vectorized split (n=1 splits only on the first dot, handling 'yahoo.co.jp' cleanly)
            splits = df[col].str.split('.', n=1, expand=True)
            
            df[f'{prefix}_emaildomain_prefix'] = splits[0].fillna('unknown')
            df[f'{prefix}_emaildomain_suffix'] = splits[1].fillna('unknown')
            
            # 2. Vectorized boolean checks (way faster than lambda)
            df[f'{prefix}_email_is_common'] = df[f'{prefix}_emaildomain_prefix'].isin(common_providers)
            
            # 3. Vectorized length calculation
            df[f'{prefix}_emaildomain_len'] = df[col].str.len().fillna(0)
    
    # Email match features
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        # True/False flags are natively understood by tree models
        df['email_domain_match'] = (df['P_emaildomain'] == df['R_emaildomain'])
        df['both_emails_missing'] = df['P_emaildomain'].isna() & df['R_emaildomain'].isna()
        
        # XOR logic for missing emails
        df['only_P_email'] = df['P_emaildomain'].notna() & df['R_emaildomain'].isna()
        df['only_R_email'] = df['P_emaildomain'].isna() & df['R_emaildomain'].notna()
        
    return df