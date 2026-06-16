import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("BEST APPROACH: Let ML models learn patterns directly")
print("="*80)

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
matches = pd.read_csv('data/matches.csv')

# Stage mapping
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
        
        hist = train[train['country'] == country].copy()
        
        if len(hist) > 0:
            feat['appearances'] = len(hist)
            feat['avg_goals'] = hist['total_goals'].mean()
            feat['max_goals'] = hist['total_goals'].max()
            feat['std_goals'] = hist['total_goals'].std() if len(hist) > 1 else 0
            feat['goals_per_match'] = hist['total_goals'].sum() / max(hist['matches_played'].sum(), 1)
            
            recent = hist.nlargest(min(3, len(hist)), 'year')
            feat['recent_goals'] = recent['total_goals'].mean()
            
            stage_vals = {'group stage': 0, 'second group stage': 1, 'round of 16': 2,
                         'quarter-finals': 3, 'semi-finals': 4, 'third-place match': 4.5,
                         'final': 5, 'final round': 5}
            hist['stage_num'] = hist['stage_reached'].map(stage_vals).fillna(0)
            
            feat['best_stage'] = hist['stage_num'].max()
            feat['avg_stage'] = hist['stage_num'].mean()
            feat['finals_count'] = (hist['stage_num'] >= 5).sum()
            feat['semifinals_count'] = (hist['stage_num'] >= 4).sum()
            
            feat['confederation'] = hist['confederation_name'].mode()[0] if len(hist) > 0 else 'Unknown'
        else:
            for key in ['appearances', 'avg_goals', 'max_goals', 'std_goals', 'goals_per_match',
                       'recent_goals', 'best_stage', 'avg_stage', 'finals_count', 'semifinals_count']:
                feat[key] = 0
            feat['confederation'] = 'Unknown'
        
        # Match stats
        team_ids = train[train['country'] == country]['team_id'].unique()
        if len(team_ids) > 0:
            team_matches = matches[(matches['home_team_id'] == team_ids[0]) | (matches['away_team_id'] == team_ids[0])]
            if len(team_matches) > 0:
                home_wins = ((team_matches['home_team_id'] == team_ids[0]) & (team_matches['home_team_win'] == 1)).sum()
                away_wins = ((team_matches['away_team_id'] == team_ids[0]) & (team_matches['away_team_win'] == 1)).sum()
                feat['win_rate'] = (home_wins + away_wins) / len(team_matches)
            else:
                feat['win_rate'] = 0
        else:
            feat['win_rate'] = 0
        
        features.append(feat)
    
    return pd.DataFrame(features)

train_features = create_features(train, is_train=True)
test_features = create_features(test, is_train=False)

# Encode
le_conf = LabelEncoder()
all_conf = pd.concat([train_features['confederation'], test_features['confederation']])
le_conf.fit(all_conf)
train_features['conf_enc'] = le_conf.transform(train_features['confederation'])
test_features['conf_enc'] = le_conf.transform(test_features['confederation'])

feature_cols = ['appearances', 'avg_goals', 'max_goals', 'std_goals', 'goals_per_match',
                'recent_goals', 'best_stage', 'avg_stage', 'finals_count', 'semifinals_count',
                'conf_enc', 'win_rate']

train_features[feature_cols] = train_features[feature_cols].fillna(0)
test_features[feature_cols] = test_features[feature_cols].fillna(0)

X_train = train_features[feature_cols].values
X_test = test_features[feature_cols].values

# ============================================================================
# TRAIN MODELS
# ============================================================================
print("\nTraining models...")

# Goals model - ensemble
gb_goals = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
rf_goals = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
ridge_goals = Ridge(alpha=1.0)

gb_goals.fit(X_train, train_features['total_goals'].values)
rf_goals.fit(X_train, train_features['total_goals'].values)
ridge_goals.fit(X_train, train_features['total_goals'].values)

pred_goals_gb = gb_goals.predict(X_test)
pred_goals_rf = rf_goals.predict(X_test)
pred_goals_ridge = ridge_goals.predict(X_test)

# Ensemble
pred_goals = 0.5 * pred_goals_gb + 0.3 * pred_goals_rf + 0.2 * pred_goals_ridge
pred_goals = np.round(pred_goals).astype(int)
pred_goals = np.maximum(pred_goals, 0)

# Stage model
le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(train_features['stage_mapped'])

gb_stage = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
rf_stage = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)

gb_stage.fit(X_train, y_stage)
rf_stage.fit(X_train, y_stage)

proba_gb = gb_stage.predict_proba(X_test)
proba_rf = rf_stage.predict_proba(X_test)
proba_ens = 0.6 * proba_gb + 0.4 * proba_rf

pred_stage = le_stage.inverse_transform(np.argmax(proba_ens, axis=1))

# ============================================================================
# MINIMAL POST-PROCESSING
# ============================================================================
print("Applying minimal calibration...")

test_features['pred_goals'] = pred_goals
test_features['pred_stage'] = pred_stage

# Ensure stage-goals consistency (only if wildly inconsistent)
for idx, row in test_features.iterrows():
    goals = row['pred_goals']
    stage = row['pred_stage']
    
    # Only adjust if clearly wrong
    if stage == 'champion' and goals < 8:
        test_features.at[idx, 'pred_goals'] = np.random.randint(10, 15)
    elif stage == 'runnerup' and goals < 7:
        test_features.at[idx, 'pred_goals'] = np.random.randint(8, 13)
    elif stage == 'sf' and goals < 5:
        test_features.at[idx, 'pred_goals'] = np.random.randint(6, 10)
    elif stage == 'qf' and goals < 4:
        test_features.at[idx, 'pred_goals'] = np.random.randint(5, 9)
    elif stage == 'roundof16' and goals < 3:
        test_features.at[idx, 'pred_goals'] = np.random.randint(3, 7)
    elif stage == 'group' and goals > 8:
        test_features.at[idx, 'pred_goals'] = np.random.randint(0, 6)

# ============================================================================
# CREATE SUBMISSION
# ============================================================================

submission = pd.DataFrame({
    'ID': test_features['ID'],
    'total_goals': test_features['pred_goals'].astype(int),
    'Target': test_features['pred_stage']
})

print("\n" + "="*80)
print("FINAL SUBMISSION")
print("="*80)

print(f"\nGoals: mean={submission['total_goals'].mean():.2f}, median={submission['total_goals'].median():.0f}")
print(f"Training: mean={train['total_goals'].mean():.2f}, median={train['total_goals'].median():.0f}")

print(f"\nStage Distribution:")
print(submission['Target'].value_counts())

print(f"\nGoals by Stage (Ours vs Training):")
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    our = submission[submission['Target'] == stage]['total_goals']
    their = train[train['stage_mapped'] == stage]['total_goals']
    if len(our) > 0 and len(their) > 0:
        print(f"  {stage:12s}: {our.mean():5.2f} vs {their.mean():5.2f}")

submission.to_csv('submission.csv', index=False)
print("\n✓ submission.csv saved!")

print("\nTop 10:")
top10 = submission.sort_values('total_goals', ascending=False).head(10)
for i, row in top10.iterrows():
    country = test[test['ID'] == row['ID']]['country'].values[0]
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")
