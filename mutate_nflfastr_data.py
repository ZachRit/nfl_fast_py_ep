import numpy as np

def make_model_mutations(df):
    df = df.copy()
    
    # Era variables
    df['era0'] = np.where(df['season'] <= 2001, 1, 0)
    df['era1'] = np.where((df['season'] >= 2002) & (df['season'] <= 2005), 1, 0)
    df['era2'] = np.where((df['season'] >= 2006) & (df['season'] <= 2013), 1, 0)
    df['era3'] = np.where((df['season'] >= 2014) & (df['season'] <= 2017), 1, 0)
    df['era4'] = np.where(df['season'] >= 2018, 1, 0)
    
    # Down variables
    df['down1'] = np.where(df['down'] == 1, 1, 0)
    df['down2'] = np.where(df['down'] == 2, 1, 0)
    df['down3'] = np.where(df['down'] == 3, 1, 0)
    df['down4'] = np.where(df['down'] == 4, 1, 0)
    
    # Home indicator
    df['home'] = np.where(df['posteam'] == df['home_team'], 1, 0)
    
    # Stadium type variables
    df['stadium_type'] = df['roof'].str.lower().str.strip()
    df['retractable'] = df['stadium_type'].str.contains('retractable').astype(float)
    df['dome'] = df['stadium_type'].str.contains('dome|closed').astype(float)
    df['outdoors'] = df['stadium_type'].str.contains('outdoor|open').astype(float)
    
    # Handle missing values
    df['retractable'] = df['retractable'].fillna(0)
    df['dome'] = df['dome'].fillna(0)
    df['outdoors'] = df['outdoors'].fillna(0)
    
    return df
