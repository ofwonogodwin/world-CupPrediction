import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FINAL OPTIMIZED MODEL - Matching training patterns precisely")
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

# Get training statistics
print("\nTraining Data Statistics:")
print(f"  Overall mean goals: {train['total_goals'].mean():.2f}")
print(f"  Overall median goals: {train['total_goals'].median():.0f}")

# Stage-specific statistics
stage_stats = {}
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    data = train[train['stage_mapped'] == stage]['total_goals']
    if len(data) > 0:
        stage_stats[stage] = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': int(data.min()),
            'max': int(data.max())
        }
        print(f"  {stage:12s}: mean={stage_stats[stage]['mean']:5.2f}, median={stage_stats[stage]['median']:4.0f}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(data_df):
    features = []
    
    for idx, row in data_df.iterrows():
        country = row['country']
        feat = {'country': country, 'ID': row['ID']}
        
        hist = train[train['country'] == country].copy()
        
        if len(hist) > 0:
            feat['appearances'] = len(hist)
            feat['avg_goals'] = hist['total_goals'].mean()
            feat['max_goals'] = hist['total_goals'].max()
            feat['goals_per_match'] = hist['total_goals'].sum() / max(hist['matches_played'].sum(), 1)
            
            recent = hist.nlargest(min(3, len(hist)), 'year')
            feat['recent_goals'] = recent['total_goals'].mean()
            
            stage_vals = {'group stage': 0, 'second group stage': 1, 'round of 16': 2,
                         'quarter-finals': 3, 'semi-finals': 4, 'third-place match': 4.5,
                         'final': 5, 'final round': 5}
            hist['stage_num'] = hist['stage_reached'].map(stage_vals).fillna(0)
            
            feat['best_stage'] = hist['stage_num'].max()
            feat['avg_stage'] = hist['stage_num'].mean()
            feat['championships'] = (hist['stage_num'] >= 5).sum()
            feat['semifinals'] = (hist['stage_num'] >= 4).sum()
            
            # Win rate from matches
            team_ids = hist['team_id'].unique()
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
        else:
            feat.update({'appearances': 0, 'avg_goals': 0, 'max_goals': 0, 'goals_per_match': 0,
                        'recent_goals': 0, 'best_stage': 0, 'avg_stage': 0, 'championships': 0,
                        'semifinals': 0, 'win_rate': 0})
        
        features.append(feat)
    
    return pd.DataFrame(features)

print("\nCreating features...")
test_features = create_features(test)

# Calculate team strength score
test_features['strength'] = (
    test_features['avg_goals'] * 0.25 +
    test_features['best_stage'] * 0.25 +
    test_features['championships'] * 0.20 +
    test_features['semifinals'] * 0.15 +
    test_features['win_rate'] * 0.10 +
    test_features['recent_goals'] * 0.05
)

# Sort by strength
test_sorted = test_features.sort_values('strength', ascending=False).reset_index(drop=True)

# ============================================================================
# ASSIGN STAGES BASED ON TOURNAMENT STRUCTURE
# ============================================================================
print("\nAssigning tournament stages...")

# For 48 teams, realistic structure:
# 1 champion, 1 runnerup, 2 SF, 4 QF, 8 R16, rest group

stages = []
for idx in range(len(test_sorted)):
    if idx == 0:
        stages.append('champion')
    elif idx == 1:
        stages.append('runnerup')
    elif idx in [2, 3]:
        stages.append('sf')
    elif idx in range(4, 8):
        stages.append('qf')
    elif idx in range(8, 16):
        stages.append('roundof16')
    else:
        stages.append('group')

test_sorted['stage'] = stages

print(f"Stage distribution: {pd.Series(stages).value_counts().to_dict()}")

# ============================================================================
# ASSIGN GOALS BASED ON STAGE AND TRAINING DISTRIBUTIONS
# ============================================================================
print("\nAssigning goals based on training patterns...")

goals = []
for idx, row in test_sorted.iterrows():
    stage = row['stage']
    
    # Sample directly from training data for this stage
    stage_train_goals = train[train['stage_mapped'] == stage]['total_goals'].values
    
    if len(stage_train_goals) > 0:
        # Random sample with preference for values closer to mean
        mean_val = stage_train_goals.mean()
        std_val = stage_train_goals.std()
        
        # Sample from normal distribution clipped to training min/max
        goal = int(np.clip(np.round(np.random.normal(mean_val, std_val * 0.75)),
                          stage_train_goals.min(), stage_train_goals.max()))
    else:
        goal = int(np.round(row['avg_goals']))
    
    goals.append(goal)

test_sorted['goals'] = goals

# Fine-tune to match overall mean if needed
current_mean = np.mean(goals)
target_mean = train['total_goals'].mean()

print(f"\nCurrent mean: {current_mean:.2f}, Target: {target_mean:.2f}")

# Only adjust if significantly off
if abs(current_mean - target_mean) > 1.0:
    # Small proportional adjustment
    adjustment = (target_mean / current_mean) ** 0.5  # Square root for gentler adjustment
    test_sorted['goals'] = (test_sorted['goals'] * adjustment).round().astype(int)
    test_sorted['goals'] = test_sorted['goals'].clip(lower=0)
else:
    # Accept the sampling as-is
    print("  (No adjustment needed - within acceptable range)")

# Sort back to original order
test_sorted = test_sorted.sort_values('ID').reset_index(drop=True)

# ============================================================================
# CREATE SUBMISSION
# ============================================================================

submission = pd.DataFrame({
    'ID': test_sorted['ID'],
    'total_goals': test_sorted['goals'].astype(int),
    'Target': test_sorted['stage']
})

print("\n" + "="*80)
print("FINAL SUBMISSION STATISTICS")
print("="*80)

print(f"\nOverall:")
print(f"  Goals mean: {submission['total_goals'].mean():.2f} (training: {train['total_goals'].mean():.2f})")
print(f"  Goals median: {submission['total_goals'].median():.0f} (training: {train['total_goals'].median():.0f})")

print(f"\nStage distribution:")
print(submission['Target'].value_counts())

print(f"\nGoals by stage (Ours vs Training):")
for stage in ['group', 'roundof16', 'qf', 'sf', 'runnerup', 'champion']:
    our = submission[submission['Target'] == stage]['total_goals']
    their = train[train['stage_mapped'] == stage]['total_goals']
    if len(our) > 0 and len(their) > 0:
        diff = our.mean() - their.mean()
        print(f"  {stage:12s}: {our.mean():5.2f} vs {their.mean():5.2f} ({diff:+5.2f})")

submission.to_csv('submission.csv', index=False)
print("\n" + "="*80)
print("✓ OPTIMIZED submission.csv saved!")
print("="*80)

print("\nTop 10 Teams:")
top10 = submission.sort_values('total_goals', ascending=False).head(10)
for i, row in top10.iterrows():
    country = test[test['ID'] == row['ID']]['country'].values[0]
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")
