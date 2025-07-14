import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
df = pd.read_csv('../../Dataset/netflix_titles.csv')
df = df[df['type'] == 'Movie']  # Solo películas
df = df.dropna(subset=['duration'])
df['duration_num'] = df['duration'].str.replace(' min', '').astype(float)
X = df[['duration_num']].fillna(0)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
plt.hist(X_pca, bins=30)
plt.title('Distribución de componentes principales')
plt.show()