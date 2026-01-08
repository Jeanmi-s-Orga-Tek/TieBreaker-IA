# Comment lancer `testLib` (Decision Tree)

Ce dossier (`testLib/`) contient une mini-pipeline **DecisionTree** qui permet :
- d'entraîner un modèle à partir des historiques ATP (CSV dans `data/atp_matches/`)
- de prédire le vainqueur d'un match entre deux joueurs (à partir de leur match le plus récent)

> Le point d'entrée CLI est : `testLib/cli/app.py`.

---

## 1) Prérequis

- Python 3.11+ (idéalement celui du projet)
- Dépendances installées :

```bash
pip install -r requirements.txt
```

- Vérifier que les données existent :
  - `data/atp_matches/atp_matches_YYYY.csv`

---

## 2) La manière recommandée : utiliser le script `run_testlib.sh`

Le plus simple est d'utiliser le script :

- `testLib/run_testlib.sh`

Il se place automatiquement à la racine du dépôt et configure `PYTHONPATH` pour éviter les erreurs du type `No module named 'testLib'`.

### Entraîner le modèle

```bash
./testLib/run_testlib.sh train --years 6 --max-depth 3
```

- `--years 6` : utilise les **6 dernières années disponibles** dans `data/atp_matches/`
- `--max-depth 3` : profondeur max de l'arbre (hyperparamètre)

Le modèle et ses métriques sont sauvegardés ici :
- `testLib/io/model.pkl`
- `testLib/io/model_meta.json`

### Prédire un match

```bash
./testLib/run_testlib.sh match --p1 "Novak Djokovic" --p2 "Carlos Alcaraz" --years 2023 2024
```

Le programme :
1. charge les matches (`testLib/data/matches.py`)
2. prend le match le plus récent pour chaque joueur
3. construit les features (`testLib/features/engineering.py`)
4. applique le modèle (`testLib/model/predictor.py`)

### Prédire en ré-entraînant juste avant

```bash
./testLib/run_testlib.sh match --p1 "Novak Djokovic" --p2 "Carlos Alcaraz" --years 2023 2024 --train --max-depth 4
```

---

## 3) (Optionnel) Changer le dossier de données

Si vos CSV ne sont pas dans `./data` :

```bash
./testLib/run_testlib.sh train --data-root /chemin/vers/data --years 6
./testLib/run_testlib.sh match --data-root /chemin/vers/data --p1 "Rafael Nadal" --p2 "Novak Djokovic"
```

Le `--data-root` doit contenir : `/chemin/vers/data/atp_matches/atp_matches_YYYY.csv`.

---

## 4) Alternative : lancer sans le script

Depuis la racine du repo, vous pouvez lancer directement le module :

```bash
python3 -m testLib.cli.app train --years 6 --max-depth 3
python3 -m testLib.cli.app match --p1 "Novak Djokovic" --p2 "Carlos Alcaraz" --years 2023 2024
```

Si vous n'êtes pas à la racine du dépôt, ajoutez le repo au `PYTHONPATH` :

```bash
PYTHONPATH=$(pwd) python3 -m testLib.cli.app train --years 6
```

---

## 5) Dépannage

### `No module named 'testLib'`
- Lancez depuis la racine du dépôt **ou** utilisez `./testLib/run_testlib.sh ...`.

### `Matches directory missing`
- Vérifiez que `data/atp_matches/` existe.
- Vérifiez qu'il contient des fichiers `atp_matches_YYYY.csv`.

### `No recent matches found for: ...`
- Le nom du joueur doit correspondre au dataset (casse/espaces ignorés mais orthographe attendue).
- Essayez un autre joueur ou élargissez les années :

```bash
./testLib/run_testlib.sh match --p1 "Novak Djokovic" --p2 "Carlos Alcaraz" --years 2019 2020 2021 2022 2023 2024
```

