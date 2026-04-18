import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create dataset
df = pd.DataFrame()
df['X1'] = [1, 2, 3, 4, 5, 6, 6, 7, 9, 9]
df['X2'] = [5, 3, 6, 8, 1, 9, 5, 8, 9, 2]
df['label'] = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]

print(df)
sns.scatterplot(x=df['X1'], y=df['X2'], hue=df['label'])

# Step 1: Initialize weights
df['weights'] = 1 / df.shape[0]
print(df)


def calculate_model_weight(error):
    return 0.5 * np.log((1 - error) / error)


def update_row_weights(row, alpha=0.423):
    if row['label'] == row['y_pred']:
        return row['weights'] * np.exp(-alpha)
    else:
        return row['weights'] * np.exp(alpha)


def create_new_dataset(df):
    indices = []
    for i in range(df.shape[0]):
        a = np.random.random()
        for index, row in df.iterrows():
            if row['cumsum_upper'] > a and a > row['cumsum_lower']:
                indices.append(index)
    return indices


# ==================== Model 1 ====================
dt1 = DecisionTreeClassifier(max_depth=1)
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values

# Step 2: Train first model
dt1.fit(X, y)
plot_tree(dt1)
plot_decision_regions(X, y, clf=dt1, legend=2)

df['y_pred'] = dt1.predict(X)
print(df)

# Step 3: Calculate model weight
alpha1 = calculate_model_weight(0.3)
print(f"alpha1: {alpha1}")

# Step 4: Update weights
df['updated_weights'] = df.apply(update_row_weights, axis=1)
print(f"Sum of updated weights: {df['updated_weights'].sum()}")

df['nomalized_weights'] = df['updated_weights'] / df['updated_weights'].sum()
print(f"Sum of normalized weights: {df['nomalized_weights'].sum()}")

df['cumsum_upper'] = np.cumsum(df['nomalized_weights'])
df['cumsum_lower'] = df['cumsum_upper'] - df['nomalized_weights']
print(df[['X1', 'X2', 'label', 'weights', 'y_pred', 'updated_weights', 'cumsum_lower', 'cumsum_upper']])

# Step 5: Create new dataset
index_values = create_new_dataset(df)
print(f"Index values: {index_values}")
second_df = df.iloc[index_values, [0, 1, 2, 3]]
print(second_df)

# ==================== Model 2 ====================
dt2 = DecisionTreeClassifier(max_depth=1)
X = second_df.iloc[:, 0:2].values
y = second_df.iloc[:, 2].values
dt2.fit(X, y)
plot_tree(dt2)
plot_decision_regions(X, y, clf=dt2, legend=2)

second_df['y_pred'] = dt2.predict(X)
print(second_df)

alpha2 = calculate_model_weight(0.1)
print(f"alpha2: {alpha2}")


def update_row_weights_2(row, alpha=1.09):
    if row['label'] == row['y_pred']:
        return row['weights'] * np.exp(-alpha)
    else:
        return row['weights'] * np.exp(alpha)


second_df['updated_weights'] = second_df.apply(update_row_weights_2, axis=1)
second_df['nomalized_weights'] = second_df['updated_weights'] / second_df['updated_weights'].sum()
print(f"Sum of normalized weights: {second_df['nomalized_weights'].sum()}")

second_df['cumsum_upper'] = np.cumsum(second_df['nomalized_weights'])
second_df['cumsum_lower'] = second_df['cumsum_upper'] - second_df['nomalized_weights']
print(second_df[['X1', 'X2', 'label', 'weights', 'y_pred', 'nomalized_weights', 'cumsum_lower', 'cumsum_upper']])

# Step 5: Create new dataset
index_values = create_new_dataset(second_df)
third_df = second_df.iloc[index_values, [0, 1, 2, 3]]
print(third_df)

# ==================== Model 3 ====================
dt3 = DecisionTreeClassifier(max_depth=1)
X = second_df.iloc[:, 0:2].values
y = second_df.iloc[:, 2].values
dt3.fit(X, y)
plot_decision_regions(X, y, clf=dt3, legend=2)

third_df['y_pred'] = dt3.predict(X)
print(third_df)

alpha3 = calculate_model_weight(0.7)
print(f"alpha1: {alpha1}, alpha2: {alpha2}, alpha3: {alpha3}")

# ==================== Prediction ====================
print("\n--- Predictions ---")

# Query 1: [1, 5]
query = np.array([1, 5]).reshape(1, 2)
pred1 = dt1.predict(query)[0]
pred2 = dt2.predict(query)[0]
pred3 = dt3.predict(query)[0]
print(f"Query [1,5]: dt1={pred1}, dt2={pred2}, dt3={pred3}")

score = alpha1 * 1 + alpha2 * 1 + alpha3 * 1
print(f"Weighted score: {score}, Sign: {np.sign(score)}")

# Query 2: [9, 9]
query = np.array([9, 9]).reshape(1, 2)
pred1 = dt1.predict(query)[0]
pred2 = dt2.predict(query)[0]
pred3 = dt3.predict(query)[0]
print(f"Query [9,9]: dt1={pred1}, dt2={pred2}, dt3={pred3}")

score = alpha1 * 1 + alpha2 * (-1) + alpha3 * (-1)
print(f"Weighted score: {score}, Sign: {np.sign(score)}")