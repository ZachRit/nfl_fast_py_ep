import pandas as pd
import numpy as np
import xgboost as xgb
from mutate_nflfastr_data import make_model_mutations
import nfl_data_py as nfl   

print('Starting...')
# Load your new play data
original_data = nfl.import_pbp_data(years=[2023], downcast=True, cache=False)

# Apply the same preprocessing steps
new_play_data = make_model_mutations(original_data)

# Select the necessary features
new_play_data = new_play_data[[
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
    'defteam_timeouts_remaining'
]].dropna()

print(new_play_data.head(10))
# One-hot encode categorical variables
# new_play_data = pd.get_dummies(new_play_data, columns=[
#     'home', 'retractable', 'dome', 'outdoors',
#     'era0', 'era1', 'era2', 'era3', 'era4',
#     'down1', 'down2', 'down3', 'down4'
# ], drop_first=True)


#print(new_play_data.head(10))
# Ensure all feature columns are numeric
new_play_data = new_play_data.apply(pd.to_numeric)
#print(new_play_data.head(10))
#features = ["half_seconds_remaining","yardline_100","ydstogo","posteam_timeouts_remaining","defteam_timeouts_remaining","home_1","dome_1.0","outdoors_1.0","era0_1","era1_1","era2_1","era3_1","era4_1","down1_1","down2_1","down3_1","down4_1"]
features = ["half_seconds_remaining","yardline_100","ydstogo","posteam_timeouts_remaining","defteam_timeouts_remaining","home","dome","outdoors","era0","era1","era2","era3","era4","down1","down2","down3","down4"]

# Match the order of columns to the training data
X_columns = features # 'X' from your training code
new_play_data = new_play_data.reindex(columns=X_columns, fill_value=0)

print(new_play_data.head(10))
new_play_data.to_csv('testing.csv', index=False)


#print(new_play_data.head(10))
# Create DMatrix for prediction
dtest = xgb.DMatrix(new_play_data)

# Load the trained model
ep_model = xgb.Booster()
#ep_model.load_model('ep_model.json')
ep_model.load_model('ed_model_v1.json')

# Make predictions
predictions = ep_model.predict(dtest)

# Point values for each scoring event
point_values = np.array([7, -7, 3, -3, 2, -2, 0])

# Calculate expected points
expected_points = predictions.dot(point_values)

# Add expected points to your DataFrame
new_play_data['expected_points'] = expected_points

# Display the results
#print(new_play_data[['expected_points']])

# Align expected_points with the original_data
# Assuming the index of new_play_data corresponds to original_data
original_data = original_data.loc[new_play_data.index]
original_data['expected_points'] = expected_points

# Display the expected points
print(original_data[['expected_points']].head(10))

# Save the original data with expected points
original_data.to_csv('original_data_with_expected_points.csv', index=False)

# Optionally, also save the transformed data
new_play_data.to_csv('new_play_data.csv', index=False)
