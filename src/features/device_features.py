import pandas as pd
import numpy as np

def create_device_features(df):
    """Clean, vectorized device feature extraction"""
    print("Creating device features...")
    
    # DeviceType
    if 'DeviceType' in df.columns:
        df['DeviceType_is_mobile'] = (df['DeviceType'] == 'mobile')
        df['DeviceType_is_desktop'] = (df['DeviceType'] == 'desktop')
    
    # DeviceInfo (Brand)
    if 'DeviceInfo' in df.columns:
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