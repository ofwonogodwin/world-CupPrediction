import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load main datasets
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_submission = pd.read_csv('SampleSubmission.csv')

# Load historical data files
matches = pd.read_csv('data/matches.csv')
goals = pd.read_csv('data/goals.csv')
teams = pd.read_csv('data/teams.csv')
players = pd.read_csv('data/players.csv')
player_appearances = pd.read_csv('data/player_appearances.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

def extract_country_code(team_id):
    """Extract country code from team_id"""
    if pd.isna(team_id):
        return None
    return team_id.split('_')[-1] if '_' in str(team_id) else team_id

# Create comprehensive features for each country
def create_features(data_df, is_train=True):
    features_list = []
    
    for idx, row in data_df.iterrows():
        country = row['country']
        feature_dict = {'country': country}
        
        if is_train:
            feature_dict['ID'] = row['ID']
            feature_dict['total_goals'] = row['total_goals']
            feature_dict['stage_reached'] = row['stage_reached']
            feature_dict['year'] = row['year']
        else:
            feature_dict['ID'] = row['ID']
        
        # Get historical data for this country
        hist_data = train[train['country'] == country] if is_train else train[train['country'] == country]
        
        if len(hist_data) > 0:
            # Basic statistics
            feature_dict['appearances'] = len(hist_data)
            feature_dict['avg_goals'] = hist_data['total_goals'].mean()
            feature_dict['max_goals'] = hist_data['total_goals'].max()
            feature_dict['min_goals'] = hist_data['total_goals'].min()
            feature_dict['std_goals'] = hist_data['total_goals'].std() if len(hist_data) > 1 else 0
            feature_dict['total_matches'] = hist_data['matches_played'].sum()
            feature_dict['avg_matches'] = hist_data['matches_played'].mean()
            
            # Goals per match
            feature_dict['goals_per_match'] = hist_data['total_goals'].sum() / max(hist_data['matches_played'].sum(), 1)
            
            # Recent performance (last 3 tournaments)
            recent = hist_data.nlargest(3, 'year')
            feature_dict['recent_avg_goals'] = recent['total_goals'].mean()
            feature_dict['recent_avg_matches'] = recent['matches_played'].mean()
            feature_dict['recent_goals_per_match'] = recent['total_goals'].sum() / max(recent['matches_played'].sum(), 1)
            
            # Stage reached features
            stage_order = {'group stage': 0, 'round of 32': 1, 'round of 16': 2, 
                          'quarter-finals': 3, 'semi-finals': 4, 'third-place match': 4.5,
                          'final': 5}
            hist_data['stage_numeric'] = hist_data['stage_reached'].map(stage_order).fillna(0)
            feature_dict['best_stage'] = hist_data['stage_numeric'].max()
            feature_dict['avg_stage'] = hist_data['stage_numeric'].mean()
            feature_dict['recent_avg_stage'] = recent['stage_numeric'].mean() if 'stage_numeric' in recent.columns else 0
            
            # Championship history
            feature_dict['championships'] = (hist_data['stage_reached'] == 'final').sum()
            feature_dict['finals'] = ((hist_data['stage_reached'] == 'final') | 
                                     (hist_data['stage_reached'] == 'third-place match')).sum()
            feature_dict['semifinals'] = (hist_data['stage_reached'].isin(['semi-finals', 'final', 'third-place match'])).sum()
            
            # Confederation
            if len(hist_data) > 0:
                feature_dict['confederation'] = hist_data['confederation_name'].mode()[0] if not hist_data['confederation_name'].mode().empty else 'Unknown'
            else:
                feature_dict['confederation'] = 'Unknown'
                
            # Time since last appearance
            if is_train and 'year' in row:
                feature_dict['years_since_last'] = row['year'] - hist_data[hist_data['year'] < row['year']]['year'].max() if len(hist_data[hist_data['year'] < row['year']]) > 0 else 100
            else:
                feature_dict['years_since_last'] = 2026 - hist_data['year'].max()
                
        else:
            # New team - set default values
            feature_dict['appearances'] = 0
            feature_dict['avg_goals'] = 0
            feature_dict['max_goals'] = 0
            feature_dict['min_goals'] = 0
            feature_dict['std_goals'] = 0
            feature_dict['total_matches'] = 0
            feature_dict['avg_matches'] = 0
            feature_dict['goals_per_match'] = 0
            feature_dict['recent_avg_goals'] = 0
            feature_dict['recent_avg_matches'] = 0
            feature_dict['recent_goals_per_match'] = 0
            feature_dict['best_stage'] = 0
            feature_dict['avg_stage'] = 0
            feature_dict['recent_avg_stage'] = 0
            feature_dict['championships'] = 0
            feature_dict['finals'] = 0
            feature_dict['semifinals'] = 0
            feature_dict['confederation'] = 'Unknown'
            feature_dict['years_since_last'] = 100
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

print("Creating training features...")
train_features = create_features(train, is_train=True)
print(f"Train features shape: {train_features.shape}")

print("Creating test features...")
test_features = create_features(test, is_train=False)
print(f"Test features shape: {test_features.shape}")

# ============================================================================
# ADVANCED FEATURES FROM MATCH DATA
# ============================================================================
print("\nAdding advanced features from match data...")

def add_match_features(df, is_train=True):
    """Add features derived from match-level data"""
    
    for idx, row in df.iterrows():
        country = row['country']
        
        # Find team_id for this country
        team_ids = train[train['country'] == country]['team_id'].unique()
        
        if len(team_ids) > 0:
            team_id = team_ids[0]
            
            # Get all matches for this team
            team_matches = matches[(matches['home_team_id'] == team_id) | 
                                   (matches['away_team_id'] == team_id)]
            
            if len(team_matches) > 0:
                # Win rate
                home_wins = ((team_matches['home_team_id'] == team_id) & 
                           (team_matches['home_team_win'] == 1)).sum()
                away_wins = ((team_matches['away_team_id'] == team_id) & 
                           (team_matches['away_team_win'] == 1)).sum()
                total_wins = home_wins + away_wins
                df.at[idx, 'win_rate'] = total_wins / len(team_matches)
                
                # Average goal difference
                home_gd = team_matches[team_matches['home_team_id'] == team_id]['home_team_score_margin'].sum()
                away_gd = team_matches[team_matches['away_team_id'] == team_id]['away_team_score_margin'].sum()
                df.at[idx, 'avg_goal_diff'] = (home_gd + away_gd) / len(team_matches)
                
                # Clean sheets
                home_clean = ((team_matches['home_team_id'] == team_id) & 
                            (team_matches['away_team_score'] == 0)).sum()
                away_clean = ((team_matches['away_team_id'] == team_id) & 
                            (team_matches['home_team_score'] == 0)).sum()
                df.at[idx, 'clean_sheet_rate'] = (home_clean + away_clean) / len(team_matches)
                
                # Recent form (last 5 matches)
                recent_matches = team_matches.sort_values('key_id', ascending=False).head(5)
                if len(recent_matches) > 0:
                    recent_home_wins = ((recent_matches['home_team_id'] == team_id) & 
                                      (recent_matches['home_team_win'] == 1)).sum()
                    recent_away_wins = ((recent_matches['away_team_id'] == team_id) & 
                                      (recent_matches['away_team_win'] == 1)).sum()
                    df.at[idx, 'recent_win_rate'] = (recent_home_wins + recent_away_wins) / len(recent_matches)
            else:
                df.at[idx, 'win_rate'] = 0
                df.at[idx, 'avg_goal_diff'] = 0
                df.at[idx, 'clean_sheet_rate'] = 0
                df.at[idx, 'recent_win_rate'] = 0
        else:
            df.at[idx, 'win_rate'] = 0
            df.at[idx, 'avg_goal_diff'] = 0
            df.at[idx, 'clean_sheet_rate'] = 0
            df.at[idx, 'recent_win_rate'] = 0
    
    return df

train_features = add_match_features(train_features, is_train=True)
test_features = add_match_features(test_features, is_train=False)

# Calculate strength score for each team
test_features['strength_score'] = (
    test_features['avg_goals'] * 0.15 +
    test_features['best_stage'] * 0.20 +
    test_features['avg_stage'] * 0.15 +
    test_features['championships'] * 0.15 +
    test_features['semifinals'] * 0.10 +
    test_features['win_rate'] * 0.15 +
    test_features['recent_avg_goals'] * 0.10
)

# ============================================================================
# PREPARE DATA FOR MODELING
# ============================================================================
print("\nPreparing data for modeling...")

# Encode confederation
le_conf = LabelEncoder()
all_confederations = pd.concat([train_features['confederation'], test_features['confederation']])
le_conf.fit(all_confederations)
train_features['confederation_encoded'] = le_conf.transform(train_features['confederation'])
test_features['confederation_encoded'] = le_conf.transform(test_features['confederation'])

# Feature columns
feature_cols = ['appearances', 'avg_goals', 'max_goals', 'min_goals', 'std_goals',
                'total_matches', 'avg_matches', 'goals_per_match',
                'recent_avg_goals', 'recent_avg_matches', 'recent_goals_per_match',
                'best_stage', 'avg_stage', 'recent_avg_stage',
                'championships', 'finals', 'semifinals',
                'confederation_encoded', 'years_since_last',
                'win_rate', 'avg_goal_diff', 'clean_sheet_rate', 'recent_win_rate']

# Fill any remaining NaN values
train_features[feature_cols] = train_features[feature_cols].fillna(0)
test_features[feature_cols] = test_features[feature_cols].fillna(0)

X_train = train_features[feature_cols].values
X_test = test_features[feature_cols].values

# ============================================================================
# MODEL 1: GOALS PREDICTION (RMSE - 60% weight)
# ============================================================================
print("\n" + "="*70)
print("TRAINING GOALS PREDICTION MODEL")
print("="*70)

y_goals = train_features['total_goals'].values

# Ensemble of models for goals
print("Training Gradient Boosting Regressor...")
gb_goals = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    subsample=0.8
)
gb_goals.fit(X_train, y_goals)

print("Training Random Forest Regressor...")
rf_goals = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_goals.fit(X_train, y_goals)

print("Training Ridge Regression...")
ridge_goals = Ridge(alpha=1.0, random_state=42)
ridge_goals.fit(X_train, y_goals)

# Ensemble predictions
pred_goals_gb = gb_goals.predict(X_test)
pred_goals_rf = rf_goals.predict(X_test)
pred_goals_ridge = ridge_goals.predict(X_test)

# Weighted ensemble
pred_goals = 0.5 * pred_goals_gb + 0.3 * pred_goals_rf + 0.2 * pred_goals_ridge

# Round and ensure non-negative
pred_goals = np.round(pred_goals).astype(int)
pred_goals = np.maximum(pred_goals, 0)

print(f"Goals predictions range: {pred_goals.min()} to {pred_goals.max()}")
print(f"Goals predictions mean: {pred_goals.mean():.2f}")

# ============================================================================
# MODEL 2: STAGE PREDICTION (F1 Score - 40% weight)
# ============================================================================
print("\n" + "="*70)
print("TRAINING STAGE PREDICTION MODEL")
print("="*70)

# Map stage names
stage_mapping = {
    'group stage': 'group',
    'round of 32': 'roundof32',
    'round of 16': 'roundof16',
    'quarter-finals': 'qf',
    'semi-finals': 'sf',
    'third-place match': 'runnerup',
    'final': 'champion'
}

train_features['stage_simplified'] = train_features['stage_reached'].map(stage_mapping)

# Encode stages
le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(train_features['stage_simplified'])

print("Training Gradient Boosting Classifier...")
gb_stage = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    subsample=0.8
)
gb_stage.fit(X_train, y_stage)

print("Training Random Forest Classifier...")
rf_stage = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
rf_stage.fit(X_train, y_stage)

# Get probability predictions
pred_stage_proba_gb = gb_stage.predict_proba(X_test)
pred_stage_proba_rf = rf_stage.predict_proba(X_test)

# Ensemble probabilities
pred_stage_proba = 0.6 * pred_stage_proba_gb + 0.4 * pred_stage_proba_rf

# Get base predictions
pred_stage = np.argmax(pred_stage_proba, axis=1)
pred_stage_names = le_stage.inverse_transform(pred_stage)

# ============================================================================
# POST-PROCESSING: ENSURE REALISTIC TOURNAMENT STRUCTURE
# ============================================================================
print("\nApplying tournament structure constraints...")

# Sort teams by strength score
test_features['base_stage'] = pred_stage_names
test_features['base_goals'] = pred_goals
test_features_sorted = test_features.sort_values('strength_score', ascending=False).reset_index(drop=True)

# Assign tournament stages realistically
# 1 champion, 1 runner-up, 2 semi-finalists, 4 quarter-finalists, 8 round of 16, rest group stage

final_stages = []
for idx, row in test_features_sorted.iterrows():
    if idx == 0:
        # Strongest team - Champion
        stage = 'champion'
    elif idx == 1:
        # Runner-up
        stage = 'runnerup'
    elif idx in [2, 3]:
        # Semi-finalists
        stage = 'sf'
    elif idx in range(4, 8):
        # Quarter-finalists
        stage = 'qf'
    elif idx in range(8, 16):
        # Round of 16
        stage = 'roundof16'
    else:
        # Group stage
        stage = 'group'
    
    final_stages.append(stage)

test_features_sorted['final_stage'] = final_stages

# Adjust goals based on stage (teams that go further score more goals)
stage_goal_adjustments = {
    'group': 1.0,
    'roundof16': 1.1,
    'qf': 1.2,
    'sf': 1.3,
    'runnerup': 1.35,
    'champion': 1.4
}

adjusted_goals = []
for idx, row in test_features_sorted.iterrows():
    base_goal = row['base_goals']
    stage = row['final_stage']
    adjustment = stage_goal_adjustments.get(stage, 1.0)
    
    # Adjust goals but keep them reasonable
    new_goals = int(np.round(base_goal * adjustment))
    
    # Ensure minimum goals for advanced stages
    if stage == 'champion':
        new_goals = max(new_goals, 10)
    elif stage == 'runnerup':
        new_goals = max(new_goals, 9)
    elif stage == 'sf':
        new_goals = max(new_goals, 7)
    elif stage == 'qf':
        new_goals = max(new_goals, 5)
    elif stage == 'roundof16':
        new_goals = max(new_goals, 4)
    
    adjusted_goals.append(new_goals)

test_features_sorted['final_goals'] = adjusted_goals

# Sort back to original order
test_features_sorted = test_features_sorted.sort_values('ID').reset_index(drop=True)

# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================
print("\n" + "="*70)
print("CREATING SUBMISSION FILE")
print("="*70)

submission = pd.DataFrame({
    'ID': test_features_sorted['ID'],
    'total_goals': test_features_sorted['final_goals'],
    'Target': test_features_sorted['final_stage']
})

print("\nFinal predictions summary:")
print(f"Total teams: {len(submission)}")
print(f"Goals range: {submission['total_goals'].min()} to {submission['total_goals'].max()}")
print(f"Goals mean: {submission['total_goals'].mean():.2f}")
print(f"\nStage distribution:")
print(submission['Target'].value_counts())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✓ Submission file created: submission.csv")

# Display top teams
print("\nTop 10 predicted teams:")
top_teams = submission.sort_values('total_goals', ascending=False).head(10)
for idx, row in top_teams.iterrows():
    country = test_features_sorted[test_features_sorted['ID'] == row['ID']]['country'].values[0]
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")

print("\n" + "="*70)
print("DONE! Good luck with the hackathon!")
print("="*70)
