import pandas as pd
import numpy as np
from pathlib import Path

n_rows = 10000

df = pd.DataFrame ({
    'employee_id': np.arange(n_rows),
    'hours_worked_today': np.random.uniform(6, 14, n_rows),
    'continous_work_hours': np.random.uniform(3, 6, n_rows),
    'tasks_completed': np.random.uniform(5, 10, n_rows),
    'break_count': np.random.randint(0, 4, n_rows),
    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Night'], n_rows),
    'previous_day_hours': np.random.uniform(5, 8, n_rows),
    'sleep_proxy': np.random.normal(7, 1.5, n_rows).clip(3, 10)
    })

df['fatigue_score'] = (
    (df['hours_worked_today'] * 0.5) +
    (df['continous_work_hours'] * 0.8) +
    (df['previous_day_hours'] * 0.2) -
    (df['break_count'] * 1.5)
)

df.loc[df['time_of_day'] == 'Night', 'fatigue_score'] *= 1.6
df.loc[df['time_of_day'] == 'Afternoon', 'fatigue_score'] += 1.5

df.loc[df['sleep_proxy'] < 5, 'fatigue_score'] *= 1.6
df.loc[df['sleep_proxy'] > 8, 'fatigue_score'] *= 0.8

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 100))
df['fatigue_risk_score'] = scaler.fit_transform(df[['fatigue_score']])

noise = np.random.normal(0, 2, n_rows)
df['fatigue_risk_score'] = (df['fatigue_risk_score'] + noise).clip(0, 100)
df['fatigue_risk_score'] = df['fatigue_risk_score'].round(1)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "fatigue_10k.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)
print("Dataset saved at:", OUTPUT_PATH)
