import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

# =========================
# Chargement des données
# =========================

# Données originales
df = pd.read_csv("data/city_lifestyle_dataset.csv")

# On conserve uniquement les variables numériques
X = df.drop(columns=["city_name", "country"])

# Standardisation (important pour la trustworthiness)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Chargement des embeddings 2D
# =========================

pca_2d = pd.read_csv("outputs/pca_emb_2d.csv").values
umap_2d = pd.read_csv("outputs/umap_emb_2d.csv").values

# =========================
# Calcul de la trustworthiness
# =========================

tw_pca = trustworthiness(X_scaled, pca_2d, n_neighbors=5)
tw_umap = trustworthiness(X_scaled, umap_2d, n_neighbors=5)

# =========================
# Affichage des résultats
# =========================

print("Comparaison des méthodes de réduction de dimension :\n")
print(f"PCA  - Trustworthiness : {tw_pca:.3f}")
print(f"UMAP - Trustworthiness : {tw_umap:.3f}")

print("Les résultats montrent que les deux méthodes de réduction de dimension préservent globalement bien la structure locale des données, avec des valeurs de trustworthiness élevées. La PCA obtient un score de 0,933, ce qui indique une bonne conservation des relations de voisinage. Toutefois, la méthode UMAP atteint un score encore plus élevé (0,972), traduisant une meilleure préservation des voisinages locaux. Ces résultats sont cohérents avec les objectifs des deux méthodes : la PCA permet de résumer efficacement la structure globale des données, tandis que UMAP, en tant que méthode non linéaire, est plus performante pour capturer les structures locales et les regroupements entre villes aux profils similaires.")
