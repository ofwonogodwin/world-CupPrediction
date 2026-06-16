import pandas as pd

# Load submission
submission = pd.read_csv('submission.csv')

print("=" * 70)
print("FIFA WORLD CUP 2026 - PREDICTION ANALYSIS")
print("=" * 70)

# Stage distribution
print("\n📊 TOURNAMENT STAGE PREDICTIONS")
print("-" * 70)
stage_counts = submission['Target'].value_counts()
for stage, count in stage_counts.items():
    print(f"{stage:15s}: {count:2d} teams")

# Top goalscorers
print("\n⚽ TOP 15 PREDICTED GOALSCORERS")
print("-" * 70)
top_scorers = submission.sort_values('total_goals', ascending=False).head(15)
for idx, row in top_scorers.iterrows():
    country = row['ID'].replace('WC-2026_', '')
    print(f"{idx+1:2d}. {country:15s} - {row['total_goals']:2d} goals - {row['Target']}")

# Goals statistics
print("\n📈 GOALS STATISTICS")
print("-" * 70)
print(f"Total goals predicted: {submission['total_goals'].sum()}")
print(f"Average goals per team: {submission['total_goals'].mean():.2f}")
print(f"Median goals: {submission['total_goals'].median():.0f}")
print(f"Range: {submission['total_goals'].min()} to {submission['total_goals'].max()}")

# By stage
print("\n🎯 AVERAGE GOALS BY STAGE")
print("-" * 70)
stage_goals = submission.groupby('Target')['total_goals'].mean().sort_values(ascending=False)
for stage, avg_goals in stage_goals.items():
    print(f"{stage:15s}: {avg_goals:.2f} goals")

# Regional distribution (approximate by country codes)
print("\n🌍 TOP TEAMS BY REGION")
print("-" * 70)

europe = ['AUT', 'BEL', 'BIH', 'HRV', 'CZE', 'ENG', 'FRA', 'DEU', 'NLD', 'NOR', 'PRT', 'SCO', 'ESP', 'SWE', 'CHE', 'TUR']
africa = ['DZA', 'CPV', 'CIV', 'COD', 'EGY', 'GHA', 'MAR', 'SEN', 'ZAF', 'TUN']
asia = ['AUS', 'IRN', 'IRQ', 'JPN', 'JOR', 'QAT', 'SAU', 'KOR', 'UZB']
south_america = ['ARG', 'BRA', 'COL', 'ECU', 'PRY', 'URY']
north_america = ['CAN', 'MEX', 'USA', 'CUW', 'HTI', 'PAN']
oceania = ['NZL']

for idx, row in submission.iterrows():
    country_code = row['ID'].replace('WC-2026_', '')
    if country_code in europe:
        region = 'Europe'
    elif country_code in africa:
        region = 'Africa'
    elif country_code in asia:
        region = 'Asia'
    elif country_code in south_america:
        region = 'South America'
    elif country_code in north_america:
        region = 'North America'
    elif country_code in oceania:
        region = 'Oceania'
    else:
        region = 'Unknown'
    submission.loc[idx, 'region'] = region

for region in ['South America', 'Europe', 'Africa', 'Asia', 'North America', 'Oceania']:
    region_data = submission[submission['region'] == region]
    if len(region_data) > 0:
        best_team = region_data.sort_values('total_goals', ascending=False).iloc[0]
        country = best_team['ID'].replace('WC-2026_', '')
        print(f"{region:20s}: {country} ({best_team['total_goals']} goals, {best_team['Target']})")

print("\n" + "=" * 70)
print("✓ SUBMISSION FILE: submission.csv is ready for upload to Zindi!")
print("=" * 70)
