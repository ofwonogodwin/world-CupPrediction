import pandas as pd
import numpy as np

print("="*80)
print("FINAL OPTIMIZATION - Calibrating predictions")
print("="*80)

# Load current submission
submission = pd.read_csv('submission.csv')
train = pd.read_csv('Train.csv')

print(f"\nCurrent submission stats:")
print(f"  Goals - Mean: {submission['total_goals'].mean():.2f}")
print(f"  Goals - Median: {submission['total_goals'].median():.0f}")
print(f"  Goals - Range: {submission['total_goals'].min()} to {submission['total_goals'].max()}")

# Training data stats
target_mean = train['total_goals'].mean()
target_median = train['total_goals'].median()

print(f"\nTarget (from training):")
print(f"  Goals - Mean: {target_mean:.2f}")
print(f"  Goals - Median: {target_median:.0f}")

# Fine-tune goals to better match distribution
# Current mean is 6.33, target is 5.56
# Apply a slight downward adjustment

adjustment_factor = target_mean / submission['total_goals'].mean()
print(f"\nAdjustment factor: {adjustment_factor:.3f}")

# Apply calibration
submission['total_goals_adjusted'] = submission['total_goals'] * adjustment_factor
submission['total_goals_adjusted'] = np.round(submission['total_goals_adjusted']).astype(int)
submission['total_goals_adjusted'] = np.maximum(submission['total_goals_adjusted'], 0)

# Ensure minimum goals for advanced stages
for idx, row in submission.iterrows():
    goals = row['total_goals_adjusted']
    stage = row['Target']
    
    if stage == 'champion':
        goals = max(goals, 10)
    elif stage == 'runnerup':
        goals = max(goals, 9)
    elif stage == 'sf':
        goals = max(goals, 7)
    elif stage == 'qf':
        goals = max(goals, 5)
    elif stage == 'roundof16':
        goals = max(goals, 4)
    elif stage == 'roundof32':
        goals = max(goals, 3)
    
    submission.at[idx, 'total_goals_adjusted'] = goals

# Update submission
submission['total_goals'] = submission['total_goals_adjusted'].astype(int)
submission = submission[['ID', 'total_goals', 'Target']]

print(f"\nAdjusted submission stats:")
print(f"  Goals - Mean: {submission['total_goals'].mean():.2f}")
print(f"  Goals - Median: {submission['total_goals'].median():.0f}")
print(f"  Goals - Range: {submission['total_goals'].min()} to {submission['total_goals'].max()}")

print(f"\nStage Distribution:")
print(submission['Target'].value_counts())

# Save final submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*80)
print("✓ FINAL OPTIMIZED submission.csv created!")
print("="*80)

# Show top teams
print("\nTop 10 Teams:")
top10 = submission.sort_values('total_goals', ascending=False).head(10)
test = pd.read_csv('Test.csv')
for idx, row in top10.iterrows():
    country = test[test['ID'].str.contains(row['ID'].split('_')[1])]['country'].values[0] if len(test[test['ID'].str.contains(row['ID'].split('_')[1])]) > 0 else row['ID']
    print(f"  {country:20s} - {row['total_goals']:2d} goals - {row['Target']}")
