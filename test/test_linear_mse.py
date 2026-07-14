#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from rutorch import Linear, MSELoss


# In[2]:


# --------------------------------------------------
# 1. Chargement des données
# --------------------------------------------------

data = pd.read_csv("./appartements_entrainement_1000.csv")

features = [
    "surface_m2",
    "qualite_localisation_0_10"
]

target = "prix_eur"


# Vérification des colonnes
required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes absentes du fichier CSV : {missing_columns}"
    )


# On conserve uniquement les colonnes nécessaires
data = data[required_columns].copy()


# Conversion en valeurs numériques
for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")


# Suppression des lignes contenant des valeurs manquantes
data = data.dropna()

print("Nombre d'exemples disponibles :", len(data))
print(data.head())


# In[3]:


# --------------------------------------------------
# 2. Séparation X / y
# --------------------------------------------------

X = data[features]
y = data[[target]]  # DataFrame 2D : forme (n, 1)


# In[4]:


# --------------------------------------------------
# 3. Séparation entraînement / validation / test
# --------------------------------------------------

# 80 % pour train + validation
# 20 % pour le test final
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1337
)

# Sur les 80 % restants :
# 75 % entraînement et 25 % validation
#
# Cela donne finalement :
# - 60 % entraînement
# - 20 % validation
# - 20 % test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=1337
)

print("\nTailles des jeux :")
print("Entraînement :", X_train.shape, y_train.shape)
print("Validation   :", X_val.shape, y_val.shape)
print("Test         :", X_test.shape, y_test.shape)


# In[5]:


# --------------------------------------------------
# 4. Conversion et normalisation
# --------------------------------------------------

# Normalisation séparée de X et de y
scaler_X = StandardScaler()
scaler_y = StandardScaler()


# fit_transform uniquement sur les données d'entraînement
np_train_X = scaler_X.fit_transform(X_train)
np_train_y = scaler_y.fit_transform(y_train)


# transform uniquement sur validation et test
np_val_X = scaler_X.transform(X_val)
np_val_y = scaler_y.transform(y_val)

np_test_X = scaler_X.transform(X_test)
np_test_y = scaler_y.transform(y_test)


# Conversion explicite en float64
np_train_X = np.asarray(np_train_X, dtype=np.float64)
np_train_y = np.asarray(np_train_y, dtype=np.float64)

np_val_X = np.asarray(np_val_X, dtype=np.float64)
np_val_y = np.asarray(np_val_y, dtype=np.float64)

np_test_X = np.asarray(np_test_X, dtype=np.float64)
np_test_y = np.asarray(np_test_y, dtype=np.float64)


print("\nFormes NumPy :")
print("Train X :", np_train_X.shape)
print("Train y :", np_train_y.shape)
print("Val X   :", np_val_X.shape)
print("Val y   :", np_val_y.shape)
print("Test X  :", np_test_X.shape)
print("Test y  :", np_test_y.shape)


# In[6]:


# --------------------------------------------------
# 5. Création du modèle
# --------------------------------------------------

np.random.seed(1337)

linear = Linear(
    input_size=np_train_X.shape[1],
    output_size=np_train_y.shape[1]
)

mse = MSELoss()


# In[7]:


# --------------------------------------------------
# 6. Entraînement
# --------------------------------------------------

epochs = 1000
learning_rate = 0.01

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_W = None
best_b = None


for epoch in range(epochs):

    # ---------------------------
    # Entraînement
    # ---------------------------

    train_pred = linear.forward(np_train_X)

    train_loss = mse.forward(
        train_pred,
        np_train_y
    )

    grad_output = mse.backward()

    # Mise à jour de W et b
    linear.backward(
        grad_output,
        learning_rate=learning_rate
    )


    # ---------------------------
    # Validation
    # ---------------------------
    # Aucun backward sur la validation

    val_pred = linear.forward(np_val_X)

    val_loss = np.mean(
        (val_pred - np_val_y) ** 2
    )


    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))


    # Sauvegarde des meilleurs paramètres
    if val_loss < best_val_loss:
        best_val_loss = float(val_loss)
        best_W = linear.W.copy()
        best_b = linear.b.copy()


    # Affichage périodique
    if epoch == 0 or (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1:4d}/{epochs} | "
            f"train loss = {train_loss:.6f} | "
            f"validation loss = {val_loss:.6f}"
        )


# In[9]:


# --------------------------------------------------
# 7. Restauration du meilleur modèle
# --------------------------------------------------

linear.W = best_W
linear.b = best_b

print(
    "\nMeilleure perte de validation normalisée :",
    best_val_loss
)


# In[10]:


# --------------------------------------------------
# 8. Évaluation sur le jeu de test
# --------------------------------------------------

test_pred_normalized = linear.forward(np_test_X)

test_loss_normalized = np.mean(
    (test_pred_normalized - np_test_y) ** 2
)

print(
    "Perte de test normalisée :",
    test_loss_normalized
)


# In[11]:


# --------------------------------------------------
# 9. Retour à l'échelle réelle des prix
# --------------------------------------------------

train_pred_eur = scaler_y.inverse_transform(
    linear.forward(np_train_X)
)

val_pred_eur = scaler_y.inverse_transform(
    linear.forward(np_val_X)
)

test_pred_eur = scaler_y.inverse_transform(
    test_pred_normalized
)


# Valeurs réelles en euros
y_train_eur = y_train.to_numpy(dtype=np.float64)
y_val_eur = y_val.to_numpy(dtype=np.float64)
y_test_eur = y_test.to_numpy(dtype=np.float64)


# In[12]:


# --------------------------------------------------
# 10. Métriques finales
# --------------------------------------------------

train_mse = mean_squared_error(
    y_train_eur,
    train_pred_eur
)

val_mse = mean_squared_error(
    y_val_eur,
    val_pred_eur
)

test_mse = mean_squared_error(
    y_test_eur,
    test_pred_eur
)

test_rmse = np.sqrt(test_mse)

test_mae = mean_absolute_error(
    y_test_eur,
    test_pred_eur
)


print("\nRésultats dans l'échelle réelle :")
print(f"MSE entraînement : {train_mse:,.2f}")
print(f"MSE validation   : {val_mse:,.2f}")
print(f"MSE test         : {test_mse:,.2f}")
print(f"RMSE test        : {test_rmse:,.2f} €")
print(f"MAE test         : {test_mae:,.2f} €")


# In[15]:


# --------------------------------------------------
# 11. Affichage de quelques prédictions
# --------------------------------------------------

results = pd.DataFrame({
    "prix_reel": y_test_eur.ravel(),
    "prix_predit": test_pred_eur.ravel()
})

results["erreur_absolue"] = np.abs(
    results["prix_reel"] - results["prix_predit"]
)

print("\nExemples de prédictions :")
print(results.head(20))


# In[ ]:




