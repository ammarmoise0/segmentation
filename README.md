# Challenge Segmentation Clients
https://www.kaggle.com/competitions/classifying-customers-into-segments/overview/description
Prédiction de segments clients (A/B/C/D) à partir de caractéristiques démographiques et comportementales.

## Le problème

Dataset de ~8000 clients avec 9 features (âge, profession, famille, etc.) et 4 segments à prédire. L'objectif est de maximiser l'accuracy sur un test set de ~2600 clients.

**Difficulté principale :** Les segments B et C se chevauchent beaucoup dans l'espace des features (observé lors de l'EDA). Les IDs sont communs entre train et test (~88%), ce qui suggère un tracking temporel des mêmes clients.

## Structure

```
.
├── Data/                          # Données : train (8068), test (2627)
├── notebooks/                     # Notebooks principaux pour l'analyse et le modeling
│   ├── data_exploration.ipynb         # Analyse exploratoire (EDA)
│   ├── comparative_modeling.ipynb     # Comparaison de 7 modèles de classification
│   └── kprototypes_clustering.ipynb   # Clustering mixte (tentative, peu concluant)
├── outputs/                  # Fichiers de soumission générés pour l'accuracy du test sur Kaggle
│
└── docs/                           # Documentation, rapports et enoncé du challenge
    ├── challenge segmentation cleint .pdf      # Enoncé original du challenge
    └── Synthèse.pdf             # synthèse de rapport final
```

## Notebooks

### 1. data_exploration.ipynb
Analyse exploratoire des données. Observations principales :
- Valeurs manquantes présentes (Work_Experience, Family_Size, Ever_Married, Var_1)
- Distribution des segments : D (28%), A (24%), C (24%), B (23%)
- Les segments B et C sont difficiles à séparer (chevauchement dans les distributions)
- 88% des IDs du test sont présents dans le train (potentiel lookback)

Visualisations avec Plotly pour interactivité.

### 2. comparative_modeling.ipynb
Test de 7 stratégies de modélisation :

1. **LightGBM Simple** : Baseline avec hyperparamètres par défaut
2. **LightGBM Lookback** : Exploitation des IDs communs train/test (hypothèse : segmentation stable)
3. **LightGBM Optuna** : Tuning automatique des hyperparamètres
4. **LightGBM Feature Selection** : Régularisation + 9 meilleures features seulement
5. **Logistic Regression** : Modèle linéaire sur données sans NaN
6. **SVM RBF** : Kernel non-linéaire
7. **Ensemble Voting** : Combinaison LightGBM + LogReg + SVM

**Résultats :** Le lookback donne 100% sur validation (car IDs identiques) mais seulement 31% sur test. L'hypothèse de segmentation stable dans le temps est fausse. Les features temporelles (âge, expérience) évoluent et changent les segments.

LGBM avec feature selection reste le meilleur compromis (33% test accuracy).

### 3. kprototypes_clustering.ipynb
Tentative de découvrir des clusters naturels avec K-Prototypes (algorithme adapté aux données mixtes numériques/catégorielles).

**Conclusion :** Échec. ARI=0.11, NMI=0.10 (proche du hasard). Les clusters découverts ne correspondent pas aux segments business. La segmentation repose probablement sur des règles métier complexes non capturables par clustering basé distance.

Pureté moyenne des clusters : 40% (il faudrait >60% pour être utile).

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate sur Windows

pip install -r requirements.txt
```

## Approche

1. EDA pour comprendre les données et identifier les difficultés (B vs C)
2. Test de plusieurs familles de modèles (tree-based, linéaires, ensemble)
3. Tentative de feature engineering via clustering (non concluant)
4. Focus sur LGBM avec feature selection et régularisation

## Résultats

Meilleure accuracy test : **~33%** (LGBM feature selection)

C'est pas terrible mais le problème est difficile :
- Les segments se chevauchent beaucoup
- Les variables disponibles semblent insuffisantes
- La segmentation business ne correspond pas à une structure naturelle dans les données

Les modèles supervisés classiques (LGBM, RF) surpassent les approches de clustering ou lookback temporel.

## Technologies

- pandas, numpy : manipulation données
- scikit-learn : ML baseline
- lightgbm : gradient boosting (meilleur modèle)
- optuna : hyperparameter tuning
- plotly : visualisations interactives
- kmodes : clustering mixte (K-Prototypes)

---

**Auteur :** Ammar MSE  
**Date :** Novembre 2025  
**Challenge :**  Data Science Interview
