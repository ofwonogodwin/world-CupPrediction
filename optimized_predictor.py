import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZED FIFA WORLD CUP 2026 PREDICTION MODEL")
print("Matching training data distributions closely")
print("="*80)

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
matches = pd.read_csv('data/matches.csv')

print(f"\nTraining: {len(train)} records")
print(f"Test: {len(test)} teams")

# Analyze training patterns
stage_map = {
    'group stage': 'group',
    'second group stage': 'group',
    'round of 16': 'roundof16',
    'quarter-finals': 'qf',
    'semi-finals': 'sf',
    'third-place match': 'runnerup',
    'final': 'champion',
    'final round': 'champion'
}
train['stage_mapped'] = train['stage_reached'].map(stage_map)

print("\nTRAINING DATA PATTERNS:")
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    data = train[train['stage_mapped'] == stage]['total_goals']
    if len(data) > 0:
        print(f"  {stage:12s}: mean={data.mean():5.2f}, median={data.median():4.0f}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(data_df, is_train=True):
    features = []
    
    for idx, row in data_df.iterrows():
        country = row['country']
        feat = {'country': country, 'ID': row['ID']}
        
        if is_train:
            feat['total_goals'] = row['total_goals']
            feat['stage_mapped'] = row['stage_mapped']
            feat['year'] = row['year']
        
        # Historical data
        hist = train[train['country'] == country].copy()
        
        if len(hist) > 0:
            feat['num_appearances'] = len(hist)
            feat['avg_goals'] = hist['total_goals'].mean()
            feat['median_goals'] = hist['total_goals'].median()
            feat['max_goals'] = hist['total_goals'].max()
            feat['std_goals'] = hist['total_goals'].std() if len(hist) > 1 else 0
            feat['total_matches'] = hist['matches_played'].sum()
            feat['goals_per_match'] = hist['total_goals'].sum() / max(hist['matches_played'].sum(), 1)
            
            # Recent performance
            recent = hist.nlargest(min(3, len(hist)), 'year')
            feat['recent_avg_goals'] = recent['total_goals'].mean()
            feat['recent_goals_per_match'] = recent['total_goals'].sum() / max(recent['matches_played'].sum(), 1)
            
            # Stage performance
            stage_vals = {'group stage': 0, 'second group stage': 1, 'round of 16': 2,
                         'quarter-finals': 3, 'semi-finals': 4, 'third-place match': 4.5,
                         'final': 5, 'final round': 5}
            hist['stage_num'] = hist['stage_reached'].map(stage_vals).fillna(0)
            
            feat['best_stage'] = hist['stage_num'].max()
            feat['avg_stage'] = hist['stage_num'].mean()
            feat['recent_avg_stage'] = recent['stage_num'].mean() if 'stage_num' in recent.columns else 0
            
            # Achievements
            feat['num_finals'] = (hist['stage_reached'].isin(['final', 'final round'])).sum()
            feat['num_semifinals'] = (hist['stage_num'] >= 4).sum()
            feat['num_knockouts'] = (hist['stage_num'] >= 2).sum()
            
            feat['confederation'] = hist['confederation_name'].mode()[0] if len(hist) > 0 else 'Unknown'
            feat['years_since_last'] = 2026 - hist['year'].max() if not is_train else (row['year'] - hist[hist['year'] < row['year']]['year'].max() if len(hist[hist['year'] < row['year']]) > 0 else 100)
        else:
            for key in ['num_appearances', 'avg_goals', 'median_goals', 'max_goals', 'std_goals',
                       'total_matches', 'goals_per_match', 'recent_avg_goals', 'recent_goals_per_match',
                       'best_stage', 'avg_stage', 'recent_avg_stage', 'num_finals', 'num_semifinals',
                       'num_knockouts', 'years_since_last']:
                feat[key] = 0
            feat['confederation'] = 'Unknown'
        
        # Match statistics
        team_ids = train[train['country'] == country]['team_id'].unique()
        if len(team_ids) > 0:
            team_id = team_ids[0]
            team_matches = matches[(matches['home_team_id'] == team_id) | (matches['away_team_id'] == team_id)]
            
            if len(team_matches) > 0:
                home_wins = ((team_matches['home_team_id'] == team_id) & (team_matches['home_team_win'] == 1)).sum()
                away_wins = ((team_matches['away_team_id'] == team_id) & (team_matches['away_team_win'] == 1)).sum()
                feat['win_rate'] = (home_wins + away_wins) / len(team_matches)
                
                home_gd = team_matches[team_matches['home_team_id'] == team_id]['home_team_score_margin'].sum()
                away_gd = team_matches[team_matches['away_team_id'] == team_id]['away_team_score_margin'].sum()
                feat['goal_difference'] = (home_gd + away_gd) / len(team_matches)
            else:
                feat['win_rate'] = 0
                feat['goal_difference'] = 0
        else:
            feat['win_rate'] = 0
            feat['goal_difference'] = 0
        
        features.append(feat)
    
    return pd.DataFrame(features)

print("\nCreating features...")
train_features = create_features(train, is_train=True)
test_features = create_features(test, is_train=False)

# Encode
le_conf = LabelEncoder()
all_conf = pd.concat([train_features['confederation'], test_features['confederation']])
le_conf.fit(all_conf)
train_features['conf_encoded'] = le_conf.transform(train_features['confederation'])
test_features['conf_encoded'] = le_conf.transform(test_features['confederation'])

feature_cols = ['num_appearances', 'avg_goals', 'median_goals', 'max_goals', 'std_goals',
                'total_matches', 'goals_per_match', 'recent_avg_goals', 'recent_goals_per_match',
                'best_stage', 'avg_stage', 'recent_avg_stage', 'num_finals', 'num_semifinals',
                'num_knockouts', 'conf_encoded', 'years_since_last', 'win_rate', 'goal_difference']

train_features[feature_cols] = train_features[feature_cols].fillna(0)
test_features[feature_cols] = test_features[feature_cols].fillna(0)

X_train = train_features[feature_cols].values
X_test = test_features[feature_cols].values
y_goals = train_features['total_goals'].values

# ============================================================================
# GOALS PREDICTION
# ============================================================================
print("\n" + "="*80)
print("TRAINING GOALS MODEL")
print("="*80)

# Use best performing model
model_goals = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=6,
    subsample=0.8,
    random_state=42
)
model_goals.fit(X_train, y_goals)

pred_goals = model_goals.predict(X_test)
pred_goals = np.maximum(pred_goals, 0)

print(f"Goals predictions: mean={pred_goals.mean():.2f}, range=[{pred_goals.min():.1f}, {pred_goals.max():.1f}]")

# ============================================================================
# STAGE PREDICTION
# ============================================================================
print("\n" + "="*80)
print("TRAINING STAGE MODEL")
print("="*80)

le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(train_features['stage_mapped'])

model_stage = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    min_samples_split=10,
    subsample=0.8,
    random_state=42
)
model_stage.fit(X_train, y_stage)

stage_proba = model_stage.predict_proba(X_test)
pred_stage_idx = np.argmax(stage_proba, axis=1)
pred_stage = le_stage.inverse_transform(pred_stage_idx)

print(f"Stage classes: {le_stage.classes_}")

# ============================================================================
# CALIBRATE TO MATCH TRAINING DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("CALIBRATING PREDICTIONS TO MATCH TRAINING PATTERNS")
print("="*80)

# Calculate team strength for ranking
test_features['strength'] = (
    test_features['avg_goals'] * 0.3 +
    test_features['best_stage'] * 0.25 +
    test_features['num_finals'] * 0.2 +
    test_features['num_semifinals'] * 0.15 +
    test_features['win_rate'] * 0.1
)

test_features['pred_goals_raw'] = pred_goals
test_features['pred_stage_raw'] = pred_stage

# Sort by strength
test_sorted = test_features.sort_values('strength', ascending=False).reset_index(drop=True)

# Get training stage-goals statistics
stage_stats = {}
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    stage_data = train[train['stage_mapped'] == stage]['total_goals']
    if len(stage_data) > 0:
        stage_stats[stage] = {
            'mean': stage_data.mean(),
            'std': stage_data.std(),
            'median': stage_data.median(),
            'min': stage_data.min(),
            'max': stage_data.max()
        }

# Assign stages based on strength ranking and match training distribution
# For 48 teams: group(majority) -> knockouts(minority based on historical success)
final_stages = []
final_goals = []

for idx in range(len(test_sorted)):
    strength = test_sorted.iloc[idx]['strength']
    base_goals = test_sorted.iloc[idx]['pred_goals_raw']
    base_stage = test_sorted.iloc[idx]['pred_stage_raw']
    
    # Assign stage based on strength percentile
    if idx == 0:  # Top team
        stage = 'champion'
    elif idx == 1:  # 2nd
        stage = 'runnerup'
    elif idx in [2, 3]:  # Top 4
        stage = 'sf'
    elif idx in range(4, 8):  # Top 8
        stage = 'qf'
    elif idx in range(8, 16):  # Top 16
        stage = 'roundof16'
    else:  # Rest go to group stage
        stage = 'group'
    
    # Calibrate goals to match training distribution for this stage
    if stage in stage_stats:
        stats = stage_stats[stage]
        # Use predicted goals but constrain to realistic range
        goals = base_goals
        
        # Apply stage-specific adjustment
        if stage == 'champion':
            goals = np.clip(goals, stats['mean'] - stats['std'], stats['mean'] + stats['std'])
            goals = max(goals, stats['median'])
        elif stage == 'runnerup':
            goals = np.clip(goals, stats['mean'] - stats['std'], stats['mean'] + stats['std'])
            goals = max(goals, stats['median'])
        elif stage == 'sf':
            goals = np.clip(goals, stats['mean'] - 1, stats['mean'] + 1)
        elif stage == 'qf':
            goals = np.clip(goals, stats['mean'] - stats['std'], stats['mean'] + stats['std'])
        elif stage == 'roundof16':
            goals = np.clip(goals, stats['mean'] - 2, stats['mean'] + 2)
        else:  # group
            goals = np.clip(goals, 0, stats['mean'] + stats['std'])
        
        goals = int(np.round(goals))
    else:
        goals = int(np.round(base_goals))
    
    final_stages.append(stage)
    final_goals.append(goals)

test_sorted['final_stage'] = final_stages
test_sorted['final_goals'] = final_goals

# Sort back to original order
test_sorted = test_sorted.sort_values('ID').reset_index(drop=True)

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print("CREATING OPTIMIZED SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'ID': test_sorted['ID'],
    'total_goals': test_sorted['final_goals'],
    'Target': test_sorted['final_stage']
})

print(f"\nFinal Statistics:")
print(f"  Goals - Mean: {submission['total_goals'].mean():.2f} (training: 5.56)")
print(f"  Goals - Median: {submission['total_goals'].median():.0f} (training: 4)")
print(f"  Goals - Range: [{submission['total_goals'].min()}, {submission['total_goals'].max()}]")

print(f"\nStage Distribution:")
print(submission['Target'].value_counts())

print(f"\nGoals by Stage:")
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    stage_data = submission[submission['Target'] == stage]['total_goals']
    if len(stage_data) > 0:
        train_data = train[train['stage_mapped'] == stage]['total_goals']
        print(f"  {stage:12s}: mean={stage_data.mean():5.2f} (train={train_data.mean():5.2f})")

submission.to_csv('submission.csv', index=False)
print("\n" + "="*80)
print("✓ OPTIMIZED submission.csv created!")
print("="*80)

# Show top teams
print("\nTop 10 Teams:")
top10 = submission.sort_values('total_goals', ascending=False).head(10)
for i, row in top10.iterrows():
    country = test[test['ID'] == row['ID']]['country'].values[0]
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")
