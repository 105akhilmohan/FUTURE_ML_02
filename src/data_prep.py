import pandas as pd

# Load dataset
df = pd.read_csv('DATA/Churn_Modelling.csv')
print(df.shape)
print(df.dtypes)

# Rename 'Exited' to 'Churn'
df.rename(columns={'Exited': 'Churn'}, inplace=True)

print(df['Churn'].value_counts(normalize=True))

# Quick checks
print(df.isna().sum())
print(df.describe(include='all'))

# Encode categorical columns
df['Geography'] = df['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Keep the target separately
y = df['Churn']
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Churn'], axis=1)