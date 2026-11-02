import pandas as pd

def create_card_features(df):
    """Create interaction features to act as proxy user IDs (Optimized)"""
    print("Creating card interaction features...")
    
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
        if col1 in df.columns and col2 in df.columns:
            new_col_name = f'{col1}_{col2}'
            # Only cast the two specific columns we need, right when we need them
            df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
            
    return df