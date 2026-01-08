<div align="center">

# 🎾 TieBreaker AI

**Prédictions intelligentes de matchs de tennis ATP/WTA**

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ligsow6/TieBreakAI?style=social)](https://github.com/ligsow6/TieBreakAI)

[Fonctionnalités](#-fonctionnalités) • [Installation](#%EF%B8%8F-installation) • [Utilisation](#-utilisation) • [Contribution](#-contribution)

</div>

---

## 📖 À propos

**TieBreaker AI** est un projet open-source de prédiction de résultats de matchs de tennis professionnels (ATP/WTA). Il combine :

- 📊 **Données historiques complètes** : plus de 50 ans de matchs ATP
- 🎯 **Système Elo adaptatif** : sensible aux surfaces (terre battue, gazon, dur, indoor)
- 🤖 **Modèles ML calibrés** : estimation précise des probabilités de victoire
- ⚡ **Interface CLI intuitive** : recherche rapide de joueurs, classements et confrontations

<div align="center">
  <img src="https://github.com/user-attachments/assets/ee6cf0ef-bd9c-48ae-818e-40cafeebf361" alt="TieBreaker AI" width="500"/>
</div>

## ✨ Fonctionnalités

- 🏆 **Consultation des classements** : historique complet des rankings ATP par joueur et par date
- ⚔️ **Recherche de confrontations** : analyse détaillée des matchs passés entre deux joueurs
- 🌍 **Filtres avancés** : par tournoi, surface, round, année
- 📈 **Base de données étendue** : matchs ATP depuis 1968, futures, challengers et qualifications inclus

## 📦 Prérequis

- Python 3.11 ou plus récent
- `pip` (fourni avec Python)
- (Optionnel) Un environnement virtuel (`venv`, `conda`, ...)
- Dépendances Python : pour l'instant `pandas` suffit à exécuter la CLI
- Jeux de données ATP déjà présents dans `data/` (sinon, placez les mêmes fichiers à cet emplacement)

## ⚙️ Installation

### Clonage du dépôt

```bash
git clone https://github.com/ligsow6/TieBreakAI.git
cd TieBreakAI
```

### Configuration de l'environnement Python

Nous recommandons l'utilisation d'un environnement virtuel pour isoler les dépendances :

```bash
# Vérification de la version Python courante
python -V

# Installation de Python 3.12.11 avec pyenv
pyenv install 3.12.11

# Configuration locale du projet
pyenv local 3.12.11

# Vérification de l'application de la nouvelle version
python -V

# Installe toutes les dépendances Python répertoriées dans requirements.txt
pip install -r requirements.txt

# Mise à jour de pip pour éviter les conflits
pip install --upgrade pip
```

### Compilation du lanceur

Avant d'utiliser la CLI, générez l'exécutable `./TieBreaker` :

```bash
# Génère le lanceur POSIX
./executable/build

# Pour nettoyer (supprimer le lanceur)
./executable/clean
```

> ⚠️ **Important** : Assurez-vous que les scripts sont exécutables avec `chmod +x executable/build executable/clean` si nécessaire.

## 🚀 Utilisation

### Commandes principales

#### Consulter un classement

```bash
./TieBreaker rank --player "Novak Djokovic"
```

Options disponibles :

- `--date YYYY-MM-DD` : classement à une date spécifique (défaut : dernier classement disponible)

#### Rechercher une confrontation

```bash
./TieBreaker match --p1 "Carlos Alcaraz" --p2 "Novak Djokovic"
```

Filtres disponibles :

- `--year YYYY` : année exacte du match
- `--tournament "Nom"` : filtre par tournoi
- `--round F|SF|QF|...` : filtre par tour (F=finale, SF=demi-finale, etc.)
- `--surface Hard|Clay|Grass|Carpet` : filtre par surface
- `--date YYYY-MM-DD` : date exacte du match
- `--all-years` : recherche sur toutes les années (plus lent)

### Exemples pratiques

```bash
# Classement de Federer au 1er janvier 2010
./TieBreaker rank --player "Roger Federer" --date 2010-01-01

# Finale de Wimbledon 2023
./TieBreaker match --p1 "Carlos Alcaraz" --p2 "Novak Djokovic" \
  --year 2023 --tournament Wimbledon --round F

# Tous les matchs sur terre battue entre Nadal et Djokovic
./TieBreaker match --p1 "Rafael Nadal" --p2 "Novak Djokovic" \
  --surface Clay --all-years
```

### Options globales

- `--data-root PATH` : chemin personnalisé vers le dossier de données (défaut : `./data`)
- `--help` : affiche l'aide détaillée

Pour plus d'informations sur une commande spécifique :

```bash
./TieBreaker rank --help
./TieBreaker match --help
```

## 🔀 Comparer deux modèles

Vous pouvez comparer deux bundles de modèles entraînés (fichiers `*.pkl` générés par `src/train_outcome.py`) sur le même jeu de matches de test.

```bash
# Exemple : comparer deux modèles et écrire un rapport JSON
./TieBreaker comparison \
  --m1 models/outcome_model_xgb_v1.pkl \
  --m2 models/outcome_model_xgb_v2.pkl \
  --l1 baseline \
  --l2 tuned \
  --report-out reports/model_comparison.json
```

Sortie typique :

```
Comparison result:
  log_loss_diff_mean: -0.0123
  log_loss_diff_ci95: [-0.0201, -0.0042]
  brier_diff_mean: -0.0038
  brier_diff_ci95: [-0.0067, -0.0011]
  accuracy_diff_mean: 0.0040
  accuracy_diff_ci95: [-0.0020, 0.0100]
  auc_diff_mean: 0.0065
  auc_diff_ci95: [0.0012, 0.0118]
```

Interprétation rapide :

- `log_loss` / `brier` plus petits = meilleur ; signe négatif pour `diff` (M1 - M2) favorise le modèle 1.
- `accuracy` / `auc` plus grands = meilleur.
- Si l'intervalle de confiance à 95 % (`ci95`) croise 0, la différence n'est pas significative avec ce bootstrap.

## 🛠️ Développement

### Architecture du projet

```text
TieBreakAI/
├── data/              # Jeux de données ATP (matchs, classements, joueurs)
├── executable/        # Scripts de build et clean
├── src/              
│   ├── main.py        # Générateur du lanceur POSIX
│   └── tiebreaker_cli.py  # Logique principale de la CLI
├── models/            # Futurs modèles ML
└── requirements.txt   # Dépendances Python
```

### Bonnes pratiques

- **Environnement virtuel** : activez-le avant chaque session (`source .venv/bin/activate`)
- **Tests** : vérifiez vos modifications avec des commandes réelles avant de commit
- **Code propre** : respectez les conventions Python (PEP 8)
- **Documentation** : commentez les fonctions complexes

### Rebuild propre

Pour repartir d'une base propre :

```bash
./executable/clean   # Supprime le lanceur
./executable/build   # Régénère le lanceur
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment participer :

1. 🍴 **Fork** le projet
2. 🌿 **Créez** une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. ✨ **Committez** vos changements (`git commit -m 'Add amazing feature'`)
4. 📤 **Pushez** vers la branche (`git push origin feature/amazing-feature`)
5. 🔃 **Ouvrez** une Pull Request

### Idées de contributions

- 🎯 Amélioration des modèles de prédiction (Elo, ML)
- 📊 Intégration de nouvelles statistiques (vitesse de service, winners, etc.)
- 🌐 Extension aux circuits WTA, ITF, Challenger
- 🖥️ Interface graphique (GUI) ou application web
- 📝 Documentation et tutoriels

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🔗 Liens utiles

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Rejoindre-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/DDPu5Vdk)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ligsow6/TieBreakAI)
[![Issues](https://img.shields.io/badge/Issues-Signaler-red?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ligsow6/TieBreakAI/issues)

</div>

---

<div align="center">

**Développé avec 🎾 par la communauté TieBreaker AI**

</div>
