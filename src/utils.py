import pandas as pd

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage using built-in Pandas downcasting"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Grab only the numeric columns to avoid messing with text
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    
    for col in numeric_cols:
        col_type = str(df[col].dtype)
        
        # Let Pandas automatically figure out the smallest size
        if 'int' in col_type:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in col_type:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        
    return df