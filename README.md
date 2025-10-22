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
## 🛠️ Documentation Développeur

### Vue d'ensemble architecturale

TieBreaker AI suit une architecture modulaire en couches :

#### Architecture en couches
```
┌─────────────────┐
│   CLI Layer     │ ← Interface utilisateur
├─────────────────┤
│ Business Logic  │ ← Règles métier (Elo, ML)
├─────────────────┤
│   Data Layer    │ ← Stockage et cache
└─────────────────┘
```

#### Modules principaux
- **ranking.py** : Système de calcul des classements ATP
- **matching.py** : Analyse des confrontations joueur vs joueur
- **data_loader.py** : Chargement et validation des données
- **cli.py** : Interface ligne de commande

### Guide de contribution

#### Processus de développement
1. 🍴 Fork le projet
2. 🌿 Créez une branche feature (`git checkout -b feature/nouvelle-fonction`)
3. 💻 Développez et testez
4. 📝 Mettez à jour la documentation
5. 🔀 Ouvrez une Pull Request

#### Standards de code
- **Python** : PEP 8, type hints, docstrings
- **Tests** : pytest obligatoire pour les nouvelles fonctionnalités
- **Commits** : messages clairs et atomiques

#### Tests
```bash
# Lancer tous les tests
pytest tests/

# Tests avec couverture
pytest --cov=src tests/
```

### API Reference

#### Classes principales

**RankingSystem**
- `get_player_ranking(player_name, date=None)` : Classement d'un joueur
- `get_top_players(limit=10, date=None)` : Top joueurs

**MatchAnalyzer**
- `analyze_head_to_head(player1, player2)` : Analyse confrontations
- `predict_match(player1, player2, surface)` : Prédiction de match

#### Exemples d'utilisation avancée

```python
from src.core.ranking import RankingSystem
from src.core.matching import MatchAnalyzer

# Analyse complète
ranking = RankingSystem()
analyzer = MatchAnalyzer()

# Classement actuel de Djokovic
rank = ranking.get_player_ranking("Novak Djokovic")

# Prédiction Nadal vs Djokovic sur terre battue
prediction = analyzer.predict_match("Rafael Nadal", "Novak Djokovic", "Clay")
```

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
# Developer Documentation

## Architecture Overview

TieBreaker AI follows a modular architecture designed for tennis match prediction:



### Core Systems

1. **Data Layer**: Historical ATP match data and player rankings
2. **CLI Interface**: Command-line interface for queries
3. **Elo System**: Adaptive ranking system for match prediction
4. **Query Engine**: Fast player and match lookup functionality

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment (recommended)

### Quick Start
```bash
# Clone and setup
git clone https://github.com/your-username/TieBreaker-IA.git
cd TieBreaker-IA
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
pip install -r requirements.txt

# Build executable
./executable/build

# Test installation
./TieBreaker --help
```

## Coding Conventions

### Python Style
- Follow PEP 8
- Use type hints for function parameters
- Maximum line length: 88 characters
- Use descriptive variable names

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- Start with component: "CLI: Add player search"
- Keep under 50 characters

### Branch Naming
- Features: `feature/description`
- Bugs: `bugfix/issue-number`
- Documentation: `docs/improvement`

## API Reference

### CLI Commands

#### Rankings Query
```bash
./TieBreaker rank --player "Player Name" [--date YYYY-MM-DD]
```

#### Match Analysis
```bash
./TieBreaker match --p1 "Player 1" --p2 "Player 2" [filters...]
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Deployment

### Local Development
```bash
# Run in development mode
python src/tiebreaker_cli.py
```

### Production Build
```bash
# Create distributable executable
./executable/build
```

## Troubleshooting

### Common Issues

1. **Permission denied on executable**
   ```bash
   chmod +x executable/build executable/clean
   ```

2. **Missing data files**
   - Ensure `data/` directory contains ATP datasets
   - Download from official ATP sources if needed

3. **Python version conflicts**
   - Use pyenv to manage Python versions
   - Recommended: Python 3.11 or 3.12

## Performance Notes

- Initial ranking queries may take 2-3 seconds due to data loading
- Subsequent queries are cached and respond in <100ms
- Memory usage scales with dataset size (currently ~500MB for full ATP history)

---

*This documentation helps new developers understand the project architecture and contribution process.*

## 🛠️ Documentation Développeur

### Vue d'ensemble de l'architecture

TieBreaker AI suit une architecture modulaire conçue pour la prédiction de matchs de tennis :

```
TieBreaker-IA/
├── data/                 # Jeux de données ATP et classements
├── src/                  # Logique applicative principale
│   ├── main.py          # Générateur du lanceur CLI
│   └── tiebreaker_cli.py # Implémentation CLI principale
├── executable/           # Scripts de build de l'exécutable
├── models/              # Stockage des modèles ML (futurs)
└── Documentation/       # Documentation du projet
```

### Systèmes principaux

1. **Couche Données** : Données historiques des matchs ATP et classements joueurs
2. **Interface CLI** : Interface en ligne de commande pour les requêtes
3. **Système Elo** : Système de classement adaptatif pour les prédictions
4. **Moteur de Requêtes** : Recherche rapide de joueurs et confrontations

### Configuration du développement

#### Prérequis
- Python 3.11+
- Git
- Environnement virtuel (recommandé)

#### Démarrage rapide
```bash
# Clonage et configuration
git clone https://github.com/username/TieBreaker-IA.git
cd TieBreaker-IA
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# Construction de l'exécutable
./executable/build

# Test d'installation
./TieBreaker --help
```

### Conventions de code

#### Style Python
- Respecter PEP 8
- Utiliser les type hints pour les paramètres de fonction
- Longueur maximale des lignes : 88 caractères
- Noms de variables descriptifs

#### Messages de commit
- Utiliser l'impératif : "Add feature" et non "Added feature"
- Commencer par le composant : "CLI: Add player search"
- Garder sous 50 caractères

#### Nommage des branches
- Fonctionnalités : `feature/description`
- Corrections : `bugfix/issue-number`
- Documentation : `docs/improvement`

### Référence API

#### Commandes CLI

##### Requête de classement
```bash
./TieBreaker rank --player "Nom Joueur" [--date YYYY-MM-DD]
```

##### Analyse de match
```bash
./TieBreaker match --p1 "Joueur 1" --p2 "Joueur 2" [filtres...]
```

### Tests

Lancer la suite de tests :
```bash
python -m pytest tests/
```

### Déploiement

#### Développement local
```bash
# Exécution en mode développement
python src/tiebreaker_cli.py
```

#### Build de production
```bash
# Créer l'exécutable distribuable
./executable/build
```

### Dépannage

#### Problèmes courants

1. **Permission refusée sur l'exécutable**
   ```bash
   chmod +x executable/build executable/clean
   ```

2. **Fichiers de données manquants**
   - S'assurer que le dossier `data/` contient les jeux de données ATP
   - Télécharger depuis les sources officielles ATP si nécessaire

3. **Conflits de version Python**
   - Utiliser pyenv pour gérer les versions Python
   - Recommandé : Python 3.11 ou 3.12

### Notes de performance

- Les premières requêtes de classement peuvent prendre 2-3 secondes en raison du chargement des données
- Les requêtes suivantes sont mises en cache et répondent en <100ms
- L'utilisation mémoire évolue avec la taille du jeu de données (actuellement ~500MB pour l'historique ATP complet)

---

*Cette documentation aide les nouveaux développeurs à comprendre l'architecture du projet et le processus de contribution.*
