import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../../Dataset/netflix_titles.csv')
movie_df = df[df['type'] == 'Movie']
movie_df = movie_df.dropna(subset=['duration'])
movie_df['duration_num'] = movie_df['duration'].str.replace(' min', 
'').astype(float)
model = IsolationForest(contamination=0.02)
movie_df['anomaly'] = model.fit_predict(movie_df[['duration_num']])
sns.histplot(data=movie_df, x='duration_num', hue='anomaly', palette='Set2', 
bins=30)
plt.title('Películas normales vs anómalas según duración')
plt.show()
print(movie_df[movie_df['anomaly'] == -1][['title', 'duration']])