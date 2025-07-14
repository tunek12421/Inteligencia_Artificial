import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
df = pd.read_csv('../../Dataset/data_rt.csv')
X = TfidfVectorizer(stop_words='english').fit_transform(df['reviews'])
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
df['cluster'] = kmeans.labels_
print(df.head())