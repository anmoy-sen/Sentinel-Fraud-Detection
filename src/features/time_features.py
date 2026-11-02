import pandas as pd
import numpy as np


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
    
    # Keep as booleans (True/False). It saves memory.
    df['Transaction_is_night'] = (df['Transaction_hour'] < 6)
    df['Transaction_is_business_hours'] = (df['Transaction_hour'] >= 9) & (df['Transaction_hour'] < 17)
    
    # Assuming days 5 and 6 represent the weekend cyclically
    df['Transaction_is_weekend'] = (df['Transaction_dow'] >= 5)
    
    return df