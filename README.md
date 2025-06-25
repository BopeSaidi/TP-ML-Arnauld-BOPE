# TP-ML-Arnauld-BOPE
# BIG DATA & Data Mining/Labo et Genie Logiciel – Régression & Classification

Ce dépôt GitHub contient deux travaux pratiques (TP) appliquant des algorithmes fondamentaux de machine learning avec Python et Jupyter Notebook.

## 📁 Contenu du dépôt

### TP 1 – Régression Linéaire : Prédiction de l'Indice de Qualité de l'Air (AQI)

- **Objectif** : Utiliser la régression linéaire pour modéliser et prédire l’AQI (Air Quality Index) à partir de données environnementales.
- **Données** : Données environnementales contenant des valeurs de SO2, NO2, PM10, etc.
- **Techniques utilisées** :
  - Nettoyage des données
  - Visualisation (matplotlib, seaborn)
  - Régression linéaire (Scikit-learn)
  - Évaluation des performances (MSE, R², etc.)

📂 Fichier : `TP-1-Regression-Linear-AQI.ipynb`

---

### TP 2 – Classification : Prédiction du Statut d'un Employé

- **Objectif** : Utiliser des algorithmes de classification pour prédire le statut (`Active`, `Resigned`, `Terminated`) d’un employé en fonction de ses caractéristiques.
- **Données** : Données RH simulées (âge, service, date d'embauche, salaire, etc.)
- **Techniques utilisées** :
  - Prétraitement et encodage des variables
  - Arbres de décision et forêts aléatoires
  - Séparation train/test
  - Matrice de confusion, classification report

📂 Fichier : `TP-2-Classification-Employee-Status-Prediction.ipynb`

---

## 🛠️ Environnement requis

- Python 3.8+
- Jupyter Notebook
- Bibliothèques :
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

🔧 Installez les dépendances :

```bash
pip install -r requirements.txt
