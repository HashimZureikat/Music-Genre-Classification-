import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
#
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data_copy = pd.read_csv(r'D:\Projects\Data Science Projects\Music Genre\untitled\Files\music_dataset_mod.csv')

# data.describe(include='all')

# data_copy.info()
data = data_copy.copy()

# unique_genres= data['Genre'].nunique
# print(unique_genres)



genre_distribution = data['Genre'].value_counts()

# Display the unique music genres and their distribution
print("Unique Music Genres and Their Distribution:")
print(genre_distribution)


plt.figure(figsize=(2, 2))
sns.barplot(x=genre_distribution.index, y=genre_distribution.values)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Music Genres')
plt.ylabel('Count')
plt.show()




data_cleaned = data.dropna()

x = data_cleaned.iloc[:, :-1]
y = data_cleaned.iloc[:,-1]

print(y.shape)
# data_cleaned.info()
# data_train, data_test =train_test_split(data_cleaned , test_size=0.2, random_state=42)
#
# data_train.shape
