import pandas as pd
import numpy as np
import nfl_data_py as nfl
import xgboost as xgb
import warnings
import pickle
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

seasons = range(1999, 2023)
pickle_file = 'pbp_data.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file):
    # Load the data from the pickle file
    with open(pickle_file, 'rb') as f:
        pbp_data = pickle.load(f)
else:
    # Import the data and save it to a pickle file
    pbp_data = nfl.import_pbp_data(years=seasons, downcast=True, cache=False)
    with open(pickle_file, 'wb') as f:
        pickle.dump(pbp_data, f)

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

pbp_data = make_model_mutations(pbp_data)

pbp_data = pbp_data.sort_values(['game_id', 'game_half', 'game_seconds_remaining'], ascending=[True, True, False])

# Create a flag for scoring plays
pbp_data['is_score'] = pbp_data[['td_team', 'field_goal_result', 'safety']].notnull().any(axis=1).astype(int)

pbp_data['Next_Score_Half'] = None
pbp_data['Drive_Score_Half'] = None

# Group by game and half
grouped = pbp_data.groupby(['game_id', 'game_half'])

next_score_list = []
drive_score_half_list = []

for name, group in grouped:
    group = group.reset_index(drop=True)
    next_score_half = ['No_Score'] * len(group)
    drive_score_half = [group['drive'].max() + 1] * len(group)
    for idx in range(len(group)):
        current_posteam = group.loc[idx, 'posteam']
        current_drive = group.loc[idx, 'drive']
        future_scores = group.loc[idx+1:, ['posteam', 'defteam', 'td_team', 'field_goal_result', 'safety', 'drive']]
        score_event = 'No_Score'
        drive_num = group['drive'].max() + 1  # Default to beyond current max drive
        for i, row in future_scores.iterrows():
            # Touchdown
            if pd.notnull(row['td_team']):
                if row['td_team'] == current_posteam:
                    score_event = 'Touchdown'
                else:
                    score_event = 'Opp_Touchdown'
                drive_num = row['drive']
                break
            # Field Goal
            elif row['field_goal_result'] == 'made':
                if row['posteam'] == current_posteam:
                    score_event = 'Field_Goal'
                else:
                    score_event = 'Opp_Field_Goal'
                drive_num = row['drive']
                break
            # Safety
            elif row['safety'] == 1:
                if row['posteam'] == current_posteam:
                    score_event = 'Safety'
                else:
                    score_event = 'Opp_Safety'
                drive_num = row['drive']
                break
        next_score_half[idx] = score_event
        drive_score_half[idx] = drive_num
    next_score_list.extend(next_score_half)
    drive_score_half_list.extend(drive_score_half)

pbp_data['Next_Score_Half'] = next_score_list
pbp_data['Drive_Score_Half'] = drive_score_half_list

# Map the Next_Score_Half to labels
label_mapping = {
    'Touchdown': 0,
    'Opp_Touchdown': 1,
    'Field_Goal': 2,
    'Opp_Field_Goal': 3,
    'Safety': 4,
    'Opp_Safety': 5,
    'No_Score': 6
}

pbp_data['label'] = pbp_data['Next_Score_Half'].map(label_mapping)

# Calculating Weights
# Calculate Drive_Score_Dist
pbp_data['Drive_Score_Dist'] = pbp_data['Drive_Score_Half'] - pbp_data['drive']

# Drive_Score_Dist Weight
max_drive_score_dist = pbp_data['Drive_Score_Dist'].max()
min_drive_score_dist = pbp_data['Drive_Score_Dist'].min()
pbp_data['Drive_Score_Dist_W'] = (max_drive_score_dist - pbp_data['Drive_Score_Dist']) / (max_drive_score_dist - min_drive_score_dist)

# Score Differential Weight
pbp_data['score_differential'] = pbp_data['score_differential'].fillna(0)
max_score_diff = pbp_data['score_differential'].abs().max()
min_score_diff = pbp_data['score_differential'].abs().min()
pbp_data['ScoreDiff_W'] = (max_score_diff - pbp_data['score_differential'].abs()) / (max_score_diff - min_score_diff)

# Total Weight
pbp_data['Total_W'] = pbp_data['Drive_Score_Dist_W'] + pbp_data['ScoreDiff_W']

# Scale Total Weight between 0 and 1
max_total_w = pbp_data['Total_W'].max()
min_total_w = pbp_data['Total_W'].min()
pbp_data['Total_W_Scaled'] = (pbp_data['Total_W'] - min_total_w) / (max_total_w - min_total_w)

# Filtering and Feature Selection
# Filter out rows with missing critical information
model_data = pbp_data[
    (~pbp_data['defteam_timeouts_remaining'].isna()) &
    (~pbp_data['posteam_timeouts_remaining'].isna()) &
    (~pbp_data['yardline_100'].isna()) &
    (~pbp_data['label'].isna())
]

# Select the relevant columns
model_data = model_data[[
    'label',
    'half_seconds_remaining',
    'yardline_100',
    'home',
    'retractable',
    'dome',
    'outdoors',
    'ydstogo',
    'era0', 'era1', 'era2', 'era3', 'era4',
    'down1', 'down2', 'down3', 'down4',
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'Total_W_Scaled'
]].dropna()

# Preparing Data for Modeling
# Ensure labels are integers
model_data['label'] = model_data['label'].astype(int)

# Separate features and target
X = model_data.drop(['label', 'Total_W_Scaled'], axis=1)
y = model_data['label']
weights = model_data['Total_W_Scaled']

# One-hot encode categorical variables if necessary
X = pd.get_dummies(X, columns=[
    'home', 'retractable', 'dome', 'outdoors',
    'era0', 'era1', 'era2', 'era3', 'era4',
    'down1', 'down2', 'down3', 'down4'
], drop_first=True)

# Ensure all feature columns are in numeric format
X = X.apply(pd.to_numeric)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y, weight=weights)

# Training the EP Model
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 7,
    'eta': 0.025,
    'gamma': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 5,
    'min_child_weight': 1,
    'seed': 2013
}

nrounds = 525

# Train the model
ep_model = xgb.train(params, dtrain, num_boost_round=nrounds)

ep_model.save_model('ep_model.json')

# Example of making predictions
# Assume you have a new dataset 'new_data' prepared in the same way as 'X'
# new_data = ...

# Prepare new data (this is just a placeholder for demonstration)
# new_data = X.iloc[:5]  # For example, take the first 5 rows

# dtest = xgb.DMatrix(new_data)
# predictions = ep_model.predict(dtest)
# The predictions array contains probabilities for each class (expected points outcomes)

# Output the model summary
print("Model training complete.")

# Show feature importance
xgb.plot_importance(ep_model)
plt.show()
