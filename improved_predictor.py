import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED FIFA WORLD CUP 2026 PREDICTION MODEL")
print("Target: Maximize combined score (60% RMSE + 40% F1)")
print("="*80)

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
matches = pd.read_csv('data/matches.csv')
goals = pd.read_csv('data/goals.csv')
teams = pd.read_csv('data/teams.csv')
tournaments = pd.read_csv('data/tournaments.csv')
team_appearances = pd.read_csv('data/team_appearances.csv')
player_appearances = pd.read_csv('data/player_appearances.csv')
squads = pd.read_csv('data/squads.csv')

print(f"\nTraining data: {len(train)} records, {train['country'].nunique()} countries")
print(f"Test data: {len(test)} teams")
print(f"Goals in training - Mean: {train['total_goals'].mean():.2f}, Median: {train['total_goals'].median():.0f}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("CREATING ADVANCED FEATURES")
print("="*80)

def create_advanced_features(data_df, is_train=True):
    """Create comprehensive features with advanced metrics"""
    features = []
    
    for idx, row in data_df.iterrows():
        country = row['country']
        feat = {'country': country, 'ID': row['ID']}
        
        if is_train:
            feat['total_goals'] = row['total_goals']
            feat['stage_reached'] = row['stage_reached']
            feat['year'] = row['year']
            feat['matches_played'] = row['matches_played']
        
        # Get historical data
        hist = train[train['country'] == country].copy()
        
        if len(hist) > 0:
            # === BASIC STATS ===
            feat['num_appearances'] = len(hist)
            feat['avg_goals'] = hist['total_goals'].mean()
            feat['median_goals'] = hist['total_goals'].median()
            feat['max_goals'] = hist['total_goals'].max()
            feat['min_goals'] = hist['total_goals'].min()
            feat['std_goals'] = hist['total_goals'].std() if len(hist) > 1 else 0
            feat['total_goals_history'] = hist['total_goals'].sum()
            
            # === MATCHES STATS ===
            feat['total_matches'] = hist['matches_played'].sum()
            feat['avg_matches'] = hist['matches_played'].mean()
            feat['max_matches'] = hist['matches_played'].max()
            
            # === GOALS PER MATCH ===
            feat['goals_per_match'] = feat['total_goals_history'] / max(feat['total_matches'], 1)
            feat['avg_goals_per_match'] = hist.apply(lambda x: x['total_goals'] / max(x['matches_played'], 1), axis=1).mean()
            
            # === STAGE PERFORMANCE ===
            stage_values = {
                'group stage': 0, 'second group stage': 1, 'round of 16': 2,
                'quarter-finals': 3, 'semi-finals': 4, 'third-place match': 4.5,
                'final': 5, 'final round': 5
            }
            hist['stage_numeric'] = hist['stage_reached'].map(stage_values).fillna(0)
            
            # === RECENCY FEATURES ===
            recent_3 = hist.nlargest(3, 'year')
            recent_5 = hist.nlargest(5, 'year')
            feat['recent3_avg_goals'] = recent_3['total_goals'].mean()
            feat['recent3_avg_matches'] = recent_3['matches_played'].mean()
            feat['recent3_goals_per_match'] = recent_3['total_goals'].sum() / max(recent_3['matches_played'].sum(), 1)
            feat['recent5_avg_goals'] = recent_5['total_goals'].mean()
            feat['recent5_goals_per_match'] = recent_5['total_goals'].sum() / max(recent_5['matches_played'].sum(), 1)
            
            feat['best_stage'] = hist['stage_numeric'].max()
            feat['avg_stage'] = hist['stage_numeric'].mean()
            feat['median_stage'] = hist['stage_numeric'].median()
            feat['recent3_avg_stage'] = recent_3['stage_numeric'].mean()
            feat['recent5_avg_stage'] = recent_5['stage_numeric'].mean()
            
            # === ACHIEVEMENT COUNTS ===
            feat['num_finals'] = (hist['stage_reached'] == 'final').sum()
            feat['num_semifinals'] = (hist['stage_reached'].isin(['semi-finals', 'third-place match', 'final', 'final round'])).sum()
            feat['num_quarterfinals'] = (hist['stage_reached'].isin(['quarter-finals', 'semi-finals', 'third-place match', 'final', 'final round'])).sum()
            feat['num_knockouts'] = (hist['stage_numeric'] >= 2).sum()
            feat['num_group_exits'] = (hist['stage_numeric'] < 2).sum()
            
            # === SUCCESS RATES ===
            feat['knockout_rate'] = feat['num_knockouts'] / len(hist)
            feat['final_rate'] = feat['num_finals'] / len(hist)
            feat['semifinal_rate'] = feat['num_semifinals'] / len(hist)
            
            # === CONFEDERATION ===
            feat['confederation'] = hist['confederation_name'].mode()[0] if len(hist['confederation_name'].mode()) > 0 else 'Unknown'
            
            # === TREND ANALYSIS ===
            if len(hist) >= 3:
                recent_trend = hist.nlargest(3, 'year')['total_goals'].values
                if len(recent_trend) == 3:
                    feat['trend_improvement'] = recent_trend[0] - recent_trend[2]
                else:
                    feat['trend_improvement'] = 0
            else:
                feat['trend_improvement'] = 0
            
            # === TIME FACTORS ===
            if is_train and 'year' in row:
                years_since = row['year'] - hist[hist['year'] < row['year']]['year'].max() if len(hist[hist['year'] < row['year']]) > 0 else 100
            else:
                years_since = 2026 - hist['year'].max()
            feat['years_since_last'] = min(years_since, 50)
            feat['last_year'] = hist['year'].max()
            feat['recency_weight'] = 1.0 / (1.0 + feat['years_since_last'] / 10.0)
            
            # === CONSISTENCY ===
            feat['cv_goals'] = feat['std_goals'] / max(feat['avg_goals'], 0.1)  # Coefficient of variation
            
        else:
            # New team - minimal history
            for key in ['num_appearances', 'avg_goals', 'median_goals', 'max_goals', 'min_goals',
                       'std_goals', 'total_goals_history', 'total_matches', 'avg_matches',
                       'max_matches', 'goals_per_match', 'avg_goals_per_match',
                       'recent3_avg_goals', 'recent3_avg_matches', 'recent3_goals_per_match',
                       'recent5_avg_goals', 'recent5_goals_per_match',
                       'best_stage', 'avg_stage', 'median_stage', 'recent3_avg_stage', 'recent5_avg_stage',
                       'num_finals', 'num_semifinals', 'num_quarterfinals', 'num_knockouts', 'num_group_exits',
                       'knockout_rate', 'final_rate', 'semifinal_rate', 'trend_improvement',
                       'years_since_last', 'recency_weight', 'cv_goals']:
                feat[key] = 0
            feat['confederation'] = 'Unknown'
            feat['last_year'] = 1900
        
        features.append(feat)
    
    df = pd.DataFrame(features)
    
    # === MATCH-LEVEL FEATURES ===
    print("Adding match-level features...")
    for idx, row in df.iterrows():
        country = row['country']
        team_ids = train[train['country'] == country]['team_id'].unique()
        
        if len(team_ids) > 0:
            team_id = team_ids[0]
            team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
            
            if len(team_matches) > 0:
                # Win statistics
                home_wins = ((team_matches['home_team_id'] == team_id) & (team_matches['home_team_win'] == 1)).sum()
                away_wins = ((team_matches['away_team_id'] == team_id) & (team_matches['away_team_win'] == 1)).sum()
                draws = (team_matches['draw'] == 1).sum()
                total_matches = len(team_matches)
                
                df.at[idx, 'win_rate'] = (home_wins + away_wins) / total_matches
                df.at[idx, 'draw_rate'] = draws / total_matches
                df.at[idx, 'loss_rate'] = 1 - df.at[idx, 'win_rate'] - df.at[idx, 'draw_rate']
                
                # Goal statistics from matches
                home_goals = team_matches[team_matches['home_team_id'] == team_id]['home_team_score'].sum()
                away_goals = team_matches[team_matches['away_team_id'] == team_id]['away_team_score'].sum()
                home_conceded = team_matches[team_matches['home_team_id'] == team_id]['away_team_score'].sum()
                away_conceded = team_matches[team_matches['away_team_id'] == team_id]['home_team_score'].sum()
                
                df.at[idx, 'total_scored'] = home_goals + away_goals
                df.at[idx, 'total_conceded'] = home_conceded + away_conceded
                df.at[idx, 'goal_difference'] = df.at[idx, 'total_scored'] - df.at[idx, 'total_conceded']
                df.at[idx, 'avg_scored_per_match'] = df.at[idx, 'total_scored'] / total_matches
                df.at[idx, 'avg_conceded_per_match'] = df.at[idx, 'total_conceded'] / total_matches
                
                # Clean sheets
                home_clean = ((team_matches['home_team_id'] == team_id) & (team_matches['away_team_score'] == 0)).sum()
                away_clean = ((team_matches['away_team_id'] == team_id) & (team_matches['home_team_score'] == 0)).sum()
                df.at[idx, 'clean_sheet_rate'] = (home_clean + away_clean) / total_matches
                
                # Recent form (last 10 matches)
                recent = team_matches.sort_values('key_id', ascending=False).head(10)
                recent_home_wins = ((recent['home_team_id'] == team_id) & (recent['home_team_win'] == 1)).sum()
                recent_away_wins = ((recent['away_team_id'] == team_id) & (recent['away_team_win'] == 1)).sum()
                df.at[idx, 'recent_win_rate'] = (recent_home_wins + recent_away_wins) / len(recent)
                
                # High scoring matches
                high_scoring = ((team_matches['home_team_id'] == team_id) & (team_matches['home_team_score'] >= 3)).sum()
                high_scoring += ((team_matches['away_team_id'] == team_id) & (team_matches['away_team_score'] >= 3)).sum()
                df.at[idx, 'high_scoring_rate'] = high_scoring / total_matches
            else:
                for key in ['win_rate', 'draw_rate', 'loss_rate', 'total_scored', 'total_conceded',
                           'goal_difference', 'avg_scored_per_match', 'avg_conceded_per_match',
                           'clean_sheet_rate', 'recent_win_rate', 'high_scoring_rate']:
                    df.at[idx, key] = 0
        else:
            for key in ['win_rate', 'draw_rate', 'loss_rate', 'total_scored', 'total_conceded',
                       'goal_difference', 'avg_scored_per_match', 'avg_conceded_per_match',
                       'clean_sheet_rate', 'recent_win_rate', 'high_scoring_rate']:
                df.at[idx, key] = 0
    
    return df

print("\nProcessing training data...")
train_features = create_advanced_features(train, is_train=True)

print("Processing test data...")
test_features = create_advanced_features(test, is_train=False)

# Encode confederation
le_conf = LabelEncoder()
all_conf = pd.concat([train_features['confederation'], test_features['confederation']])
le_conf.fit(all_conf)
train_features['confederation_encoded'] = le_conf.transform(train_features['confederation'])
test_features['confederation_encoded'] = le_conf.transform(test_features['confederation'])

# Feature columns for modeling
feature_cols = [
    'num_appearances', 'avg_goals', 'median_goals', 'max_goals', 'std_goals',
    'total_matches', 'avg_matches', 'goals_per_match', 'avg_goals_per_match',
    'recent3_avg_goals', 'recent3_goals_per_match', 'recent5_avg_goals', 'recent5_goals_per_match',
    'best_stage', 'avg_stage', 'median_stage', 'recent3_avg_stage', 'recent5_avg_stage',
    'num_finals', 'num_semifinals', 'num_quarterfinals', 'knockout_rate', 'final_rate', 'semifinal_rate',
    'confederation_encoded', 'years_since_last', 'recency_weight', 'trend_improvement', 'cv_goals',
    'win_rate', 'draw_rate', 'goal_difference', 'avg_scored_per_match', 'avg_conceded_per_match',
    'clean_sheet_rate', 'recent_win_rate', 'high_scoring_rate'
]

train_features[feature_cols] = train_features[feature_cols].fillna(0)
test_features[feature_cols] = test_features[feature_cols].fillna(0)

X_train = train_features[feature_cols].values
X_test = test_features[feature_cols].values
y_goals = train_features['total_goals'].values

print(f"\nFeature matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")

# ============================================================================
# GOALS PREDICTION MODEL (60% weight - RMSE)
# ============================================================================
print("\n" + "="*80)
print("TRAINING GOALS PREDICTION MODELS WITH CROSS-VALIDATION")
print("="*80)

# Multiple models with different strengths
models_goals = {
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                   min_samples_split=8, min_samples_leaf=4,
                                                   subsample=0.8, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=8,
                                         min_samples_leaf=4, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=8,
                                      min_samples_leaf=4, random_state=42),
    'Ridge': Ridge(alpha=0.5, random_state=42)
}

trained_models_goals = {}
for name, model in models_goals.items():
    model.fit(X_train, y_goals)
    trained_models_goals[name] = model
    cv_scores = cross_val_score(model, X_train, y_goals, cv=5, scoring='neg_root_mean_squared_error')
    print(f"{name:20s} - CV RMSE: {-cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Ensemble predictions with optimized weights
pred_gb = trained_models_goals['GradientBoosting'].predict(X_test)
pred_rf = trained_models_goals['RandomForest'].predict(X_test)
pred_et = trained_models_goals['ExtraTrees'].predict(X_test)
pred_ridge = trained_models_goals['Ridge'].predict(X_test)

# Weighted ensemble (tuned for better RMSE)
pred_goals = 0.4 * pred_gb + 0.3 * pred_rf + 0.2 * pred_et + 0.1 * pred_ridge

# Post-processing: calibrate to match training distribution
pred_goals = np.round(pred_goals).astype(int)
pred_goals = np.maximum(pred_goals, 0)

# Adjust predictions to match historical mean better
current_mean = pred_goals.mean()
target_mean = train['total_goals'].mean()
adjustment_factor = target_mean / max(current_mean, 1)

if abs(adjustment_factor - 1.0) > 0.1:  # If significantly different
    pred_goals = np.round(pred_goals * adjustment_factor).astype(int)
    pred_goals = np.maximum(pred_goals, 0)

print(f"\nGoals Predictions:")
print(f"  Range: {pred_goals.min()} to {pred_goals.max()}")
print(f"  Mean: {pred_goals.mean():.2f} (target: {target_mean:.2f})")
print(f"  Median: {np.median(pred_goals):.0f}")

# ============================================================================
# STAGE PREDICTION MODEL (40% weight - F1 Score)
# ============================================================================
print("\n" + "="*80)
print("TRAINING STAGE PREDICTION MODELS")
print("="*80)

# Map training stages to submission format
stage_mapping = {
    'group stage': 'group',
    'second group stage': 'group',
    'round of 16': 'roundof16',
    'quarter-finals': 'qf',
    'semi-finals': 'sf',
    'third-place match': 'runnerup',
    'final': 'champion',
    'final round': 'champion'
}

train_features['stage_mapped'] = train_features['stage_reached'].map(stage_mapping)

le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(train_features['stage_mapped'])

print(f"Stage classes: {le_stage.classes_}")
print(f"Stage distribution in training: {pd.Series(train_features['stage_mapped']).value_counts().to_dict()}")

# Train multiple classifiers
models_stage = {
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                    min_samples_split=8, subsample=0.8, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=8,
                                          class_weight='balanced', random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_split=8,
                                       class_weight='balanced', random_state=42)
}

trained_models_stage = {}
for name, model in models_stage.items():
    model.fit(X_train, y_stage)
    trained_models_stage[name] = model
    print(f"{name:20s} - trained")

# Ensemble probabilities
proba_gb = trained_models_stage['GradientBoosting'].predict_proba(X_test)
proba_rf = trained_models_stage['RandomForest'].predict_proba(X_test)
proba_et = trained_models_stage['ExtraTrees'].predict_proba(X_test)

proba_ensemble = 0.45 * proba_gb + 0.35 * proba_rf + 0.2 * proba_et

# Calculate team strength scores
test_features['strength_score'] = (
    test_features['avg_goals'] * 0.15 +
    test_features['recent3_avg_goals'] * 0.15 +
    test_features['best_stage'] * 0.20 +
    test_features['avg_stage'] * 0.10 +
    test_features['num_finals'] * 0.15 +
    test_features['num_semifinals'] * 0.10 +
    test_features['win_rate'] * 0.10 +
    test_features['recent_win_rate'] * 0.05
)

test_features['pred_goals'] = pred_goals
test_features['base_stage_idx'] = np.argmax(proba_ensemble, axis=1)
test_features['base_stage'] = le_stage.inverse_transform(test_features['base_stage_idx'])

# Sort by strength for realistic tournament structure
test_sorted = test_features.sort_values('strength_score', ascending=False).reset_index(drop=True)

# Assign stages based on 2026 World Cup structure (48 teams)
# Group stage: 48 teams, 32 advance
# Round of 32 (16 games) -> Round of 16 (8 games) -> QF (4 games) -> SF (2 games) -> Final
final_stages = []
final_goals = []

for idx in range(len(test_sorted)):
    base_goals = test_sorted.iloc[idx]['pred_goals']
    strength = test_sorted.iloc[idx]['strength_score']
    
    # Assign stages based on ranking
    if idx == 0:
        stage = 'champion'
        goal_mult = 1.5
        min_goals = 12
    elif idx == 1:
        stage = 'runnerup'
        goal_mult = 1.4
        min_goals = 10
    elif idx in [2, 3]:
        stage = 'sf'
        goal_mult = 1.3
        min_goals = 8
    elif idx in range(4, 8):
        stage = 'qf'
        goal_mult = 1.2
        min_goals = 6
    elif idx in range(8, 16):
        stage = 'roundof16'
        goal_mult = 1.1
        min_goals = 4
    elif idx in range(16, 32):
        stage = 'roundof32'
        goal_mult = 1.05
        min_goals = 3
    else:
        stage = 'group'
        goal_mult = 0.95
        min_goals = 0
    
    # Adjust goals based on stage
    adjusted_goals = int(np.round(base_goals * goal_mult))
    adjusted_goals = max(adjusted_goals, min_goals)
    
    final_stages.append(stage)
    final_goals.append(adjusted_goals)

test_sorted['final_stage'] = final_stages
test_sorted['final_goals'] = final_goals

# Sort back to original order
test_sorted = test_sorted.sort_values('ID').reset_index(drop=True)

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("CREATING IMPROVED SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'ID': test_sorted['ID'],
    'total_goals': test_sorted['final_goals'],
    'Target': test_sorted['final_stage']
})

print(f"\nFinal Statistics:")
print(f"  Total teams: {len(submission)}")
print(f"  Goals - Mean: {submission['total_goals'].mean():.2f}, Median: {submission['total_goals'].median():.0f}")
print(f"  Goals - Range: {submission['total_goals'].min()} to {submission['total_goals'].max()}")
print(f"\nStage Distribution:")
print(submission['Target'].value_counts())

# Save
submission.to_csv('submission.csv', index=False)
print("\n" + "="*80)
print("✓ IMPROVED submission.csv created!")
print("="*80)

# Show top predictions
print("\nTop 15 Teams:")
top15 = submission.sort_values('total_goals', ascending=False).head(15)
for i, row in top15.iterrows():
    country = test_sorted[test_sorted['ID'] == row['ID']]['country'].values[0]
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")
