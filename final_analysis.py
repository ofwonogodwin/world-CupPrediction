import pandas as pd
import numpy as np

print('='*80)
print('FINAL SUBMISSION ANALYSIS')
print('='*80)

sub = pd.read_csv('submission.csv')
train = pd.read_csv('Train.csv')

print('\n📊 GOALS STATISTICS')
print('-'*80)
print(f'Training data mean:    {train["total_goals"].mean():.2f}')
print(f'Submission mean:       {sub["total_goals"].mean():.2f}  <- CLOSE MATCH!')
print(f'Difference:            {abs(sub["total_goals"].mean() - train["total_goals"].mean()):.2f}')
print(f'')
print(f'Training median:       {train["total_goals"].median():.0f}')
print(f'Submission median:     {sub["total_goals"].median():.0f}')
print(f'')
print(f'Training std:          {train["total_goals"].std():.2f}')
print(f'Submission std:        {sub["total_goals"].std():.2f}')

print('\n🏆 STAGE DISTRIBUTION')
print('-'*80)
stage_dist = sub['Target'].value_counts().sort_index()
for stage, count in stage_dist.items():
    pct = count / len(sub) * 100
    print(f'{stage:15s}: {count:2d} teams ({pct:5.1f}%)')

print('\n⚽ GOALS BY STAGE')
print('-'*80)
stage_goals = sub.groupby('Target')['total_goals'].agg(['mean', 'min', 'max'])
stage_goals = stage_goals.sort_values('mean', ascending=False)
for stage, row in stage_goals.iterrows():
    print(f'{stage:15s}: {row["mean"]:5.1f} avg  (range: {int(row["min"]):2d}-{int(row["max"]):2d})')

print('\n🌟 TOP 10 TEAMS')
print('-'*80)
test = pd.read_csv('Test.csv')
top10 = sub.sort_values('total_goals', ascending=False).head(10)
for i, (idx, row) in enumerate(top10.iterrows(), 1):
    country_code = row['ID'].split('_')[1]
    country = test[test['ID'].str.contains(country_code)]['country'].values[0]
    print(f'{i:2d}. {country:20s} - {row["total_goals"]:2d} goals - {row["Target"]}')

print('\n' + '='*80)
print('✅ SUBMISSION READY FOR UPLOAD!')
print('='*80)
