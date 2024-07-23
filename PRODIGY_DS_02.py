import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
titanic_data.head()

# Get the basic information and summary statistics
titanic_data.info()
titanic_data.describe()

# Check for missing values
missing_values = titanic_data.isnull().sum()

# Display columns with missing values
missing_values[missing_values > 0]

# Fill missing 'Age' values with the median age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())

titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# Drop the 'Cabin' column due to too many missing values
titanic_data.drop('Cabin', axis=1, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of Age
plt.figure(figsize=(10, 5))
sns.histplot(titanic_data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count (0 = Not Survived, 1 = Survived)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Plot survival rate by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Plot survival rate by passenger class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Plot survival rate by embarkation port
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival Rate by Embarkation Port')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

# Compute the correlation matrix
numeric_cols = titanic_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Pair plot of selected features
sns.pairplot(titanic_data[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()
