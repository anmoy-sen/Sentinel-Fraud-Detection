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



def create_time_features(df):
    """Clean and optimized time-based feature extraction"""
    print("Creating time features...")
    
    # Calculate base conversions once to save computation time
    hours_total = np.floor(df['TransactionDT'] / 3600)
    days_total = np.floor(df['TransactionDT'] / 86400) # 86400 seconds in a day
    
    # Core cyclical features
    df['Transaction_hour'] = hours_total % 24
    df['Transaction_dow'] = days_total % 7
    df['Transaction_dom'] = days_total % 30
    
    # Time of day binning (labels=False outputs 0, 1, 2, 3 natively)
    df['Transaction_time_of_day'] = pd.cut(df['Transaction_hour'], 
                                           bins=[-1, 6, 12, 18, 24], 
                                           labels=False)
    
    # Keep as booleans (True/False) - XGBoost handles this and it saves memory
    df['Transaction_is_night'] = (df['Transaction_hour'] < 6)
    df['Transaction_is_business_hours'] = (df['Transaction_hour'] >= 9) & (df['Transaction_hour'] < 17)
    
    # Assuming days 5 and 6 represent the weekend cyclically
    df['Transaction_is_weekend'] = (df['Transaction_dow'] >= 5)
    
    return df




def create_card_features(df):
    """Create interaction features to act as proxy user IDs"""
    print("Creating card interaction features...")
    
    # We use 'nan' as a string so the model knows it's a missing interaction, 
    # rather than just failing the concatenation
    df_str = df.astype(str)
    
    # Define the pairs you want to combine
    interaction_pairs = [
        ('card1', 'card2'),
        ('card4', 'card6'),
        ('card1', 'addr1'),
        ('card1', 'addr2'),
        ('card2', 'addr1'),
        ('card1', 'ProductCD'),
        ('card2', 'ProductCD'),
        ('card1', 'Transaction_hour'),
        ('card1', 'Transaction_dow')
    ]
    
    for col1, col2 in interaction_pairs:
        # Only combine if both columns actually exist in the dataframe
        if col1 in df.columns and col2 in df.columns:
            new_col_name = f'{col1}_{col2}'
            df[new_col_name] = df_str[col1] + '_' + df_str[col2]
            
    return df




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



def create_device_features(df):
    """Clean, vectorized device feature extraction"""
    print("Creating device features...")
    
    # DeviceType
    if 'DeviceType' in df.columns:
        df['DeviceType_is_mobile'] = (df['DeviceType'] == 'mobile')
        df['DeviceType_is_desktop'] = (df['DeviceType'] == 'desktop')
    
    # DeviceInfo (Brand)
    if 'DeviceInfo' in df.columns:
        # Convert to lowercase string once to save time
        dev_str = df['DeviceInfo'].astype(str).str.lower()
        
        # Vectorized split and extract
        df['Device_brand'] = dev_str.str.split('/').str[0].str.split().str[0]
        df['DeviceInfo_len'] = df['DeviceInfo'].str.len().fillna(0)
        
        # Vectorized string matching
        df['Device_is_Samsung'] = dev_str.str.contains('samsung', na=False)
        df['Device_is_iOS'] = dev_str.str.contains('ios|iphone|ipad', na=False)
        df['Device_is_Windows'] = dev_str.str.contains('windows', na=False)
    
    # Browser features (id_31)
    if 'id_31' in df.columns:
        browser_str = df['id_31'].astype(str).str.lower()
        df['Browser'] = browser_str.str.split().str[0]
        
        df['Browser_is_chrome'] = browser_str.str.contains('chrome', na=False)
        df['Browser_is_safari'] = browser_str.str.contains('safari', na=False)
        df['Browser_is_firefox'] = browser_str.str.contains('firefox', na=False)
        df['Browser_is_edge'] = browser_str.str.contains('edge', na=False)
    
    # OS features (id_30)
    if 'id_30' in df.columns:
        os_str = df['id_30'].astype(str).str.lower()
        df['OS'] = os_str.str.split().str[0]
        
        df['OS_is_Windows'] = os_str.str.contains('windows', na=False)
        df['OS_is_Mac'] = os_str.str.contains('mac', na=False)
        df['OS_is_iOS'] = os_str.str.contains('ios', na=False)
        df['OS_is_Android'] = os_str.str.contains('android', na=False)
    
    # Screen resolution (id_33)
    if 'id_33' in df.columns:
        # Vectorized split into two columns (width and height)
        res_split = df['id_33'].str.split('x', expand=True)
        
        # Convert to numbers. errors='coerce' turns bad data (like 'unknown') into NaN
        df['Screen_width'] = pd.to_numeric(res_split[0], errors='coerce').fillna(-1)
        df['Screen_height'] = pd.to_numeric(res_split[1], errors='coerce').fillna(-1)
        
        # Vectorized math
        df['Screen_area'] = df['Screen_width'] * df['Screen_height']
        
        # np.where is an instant if/else statement for entire columns.
        # It prevents division by zero errors where height might be 0 or -1.
        df['Screen_aspect_ratio'] = np.where(df['Screen_height'] > 0, 
                                             df['Screen_width'] / df['Screen_height'], 
                                             -1)
    
    return df




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
        'v2_behav': [f'V{i}' for i in range(12, 35)],    # V12 to V34 (Corrected from Kaggle!)
        'v3_hist': [f'V{i}' for i in range(35, 53)],     # V35 to V52
        'v4_card': [f'V{i}' for i in range(53, 75)],     # V53 to V74
        'v5_geo': [f'V{i}' for i in range(75, 95)],      # V75 to V94
        'v6_item': [f'V{i}' for i in range(95, 138)],    # V95 to V137
        'v7_dev1': [f'V{i}' for i in range(138, 167)],   # V138 to V166 (Combined contiguous block)
        'v8_dev2': [f'V{i}' for i in range(167, 217)],   # V167 to V216 
        'v9_net1': [f'V{i}' for i in range(217, 279)],   # V217 to V278 
        'v10_net2': [f'V{i}' for i in range(279, 322)],  # V279 to V321 
        'v11_stat': [f'V{i}' for i in range(322, 340)],  # V322 to V339
    }
    
    # Dictionary to hold new columns to prevent DataFrame fragmentation
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
        
    # Concatenate everything at once for maximum memory efficiency
    new_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, new_df], axis=1)





def create_c_d_features(df):
    """Clean, fragmentation-safe feature extraction for Counts and Deltas"""
    print("Creating C and D column features...")
    
    # Store new features here to prevent fragmentation
    new_features = {}
    
    # -------------------------
    # C Columns (Counts)
    # -------------------------
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
            
    # -------------------------
    # D Columns (Time Deltas)
    # -------------------------
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
                
    # Concatenate all new features efficiently
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)
        
    return df






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
        # Create boolean masks (Memory efficient)
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







def create_id_features(df):
    """Clean, fragmentation-safe extraction for Identity columns"""
    print("Creating ID features...")
    
    new_features = {}
    
    # -------------------------
    # Numerical IDs (id_01 to id_11)
    # -------------------------
    id_num_cols = [f'id_0{i}' for i in range(1, 10)] + ['id_10', 'id_11']
    existing_id_num = [col for col in id_num_cols if col in df.columns]
    
    if existing_id_num:
        subset_num = df[existing_id_num]
        new_features['id_num_nan_count'] = subset_num.isna().sum(axis=1)
        new_features['id_num_mean'] = subset_num.mean(axis=1)
        new_features['id_num_std'] = subset_num.std(axis=1)
        
    # -------------------------
    # Categorical Statuses (New vs Found)
    # -------------------------
    # Using dictionary mapping to avoid repeating if-statements
    status_cols = ['id_12', 'id_15', 'id_16', 'id_28', 'id_29']
    
    for col in status_cols:
        if col in df.columns:
            new_features[f'{col}_isFound'] = (df[col] == 'Found')
            # Only some columns track 'New', but checking it dynamically is safe
            if 'New' in df[col].unique():
                new_features[f'{col}_isNew'] = (df[col] == 'New')

    # -------------------------
    # String Parsing (id_34)
    # -------------------------
    if 'id_34' in df.columns:
        # Original used a slow lambda. Vectorized split is instantly fast in C.
        # Format is usually 'match_status:1' or 'match_status:2'
        split_vals = df['id_34'].str.split(':', expand=True)
        # If the split worked (column 1 exists), convert to float, else fill -1
        if 1 in split_vals.columns:
            new_features['id_34_match'] = pd.to_numeric(split_vals[1], errors='coerce').fillna(-1)
        else:
            new_features['id_34_match'] = -1

    # -------------------------
    # Boolean Flags (id_36, 37, 38)
    # -------------------------
    for col in ['id_36', 'id_37', 'id_38']:
        if col in df.columns:
            new_features[f'{col}_isT'] = (df[col] == 'T')
            
    # Concatenate safely
    if new_features:
        new_df = pd.DataFrame(new_features, index=df.index)
        return pd.concat([df, new_df], axis=1)
        
    return df