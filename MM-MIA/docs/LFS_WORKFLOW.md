# Git LFS — Workflow équipe MM-MIA

Ce projet utilise **Git LFS** (Large File Storage) pour gérer les fichiers volumineux : datasets DICOM, archives zip d'images, modèles entraînés (.pth, .ckpt), images preprocessed.

Backend LFS : **GitLab Télécom Paris** (`gitlab.enst.fr`).
Le miroir GitHub ne reçoit que les pointeurs LFS (pas les blobs).

---

## 🚀 Setup initial (une fois par machine)

### 1. Installer Git LFS

```bash
# Debian / Ubuntu
sudo apt install git-lfs

# macOS
brew install git-lfs

# Windows : télécharger depuis https://git-lfs.com/
```

### 2. Activer LFS pour ton utilisateur

```bash
git lfs install   # ajoute les hooks LFS dans ~/.gitconfig
```

À faire **une seule fois** par machine. Inutile de répéter pour chaque repo.

### 3. Cloner le repo

```bash
git clone git@gitlab.enst.fr:proj104/2025-2026/MM-MIA.git
cd MM-MIA
```

`git clone` télécharge **automatiquement** les fichiers LFS si LFS est installé. Si tu vois `data/image_classification/train.zip` faisant seulement quelques KB avec du texte commençant par `version https://git-lfs.github.com/...` → LFS n'a pas téléchargé, lance :

```bash
git lfs pull
```

---

## 📥 Récupérer les fichiers LFS

### Tout télécharger
```bash
git lfs pull
```
⚠️ Peut prendre du temps et bande passante (~1 Gb actuellement).

### Filtré (recommandé)
```bash
# Seulement le dataset text
git lfs pull --include="data/text_classification/**" --include="data/shared/**"

# Seulement les DICOM
git lfs pull --include="data/dicom/**"

# Seulement les modèles
git lfs pull --include="models/**"

# Tout SAUF les zips lourds (gain énorme si tu n'entraînes pas en image)
git lfs pull --exclude="data/image_classification/*.zip"
```

### Voir ce qu'il y a en LFS
```bash
git lfs ls-files                # Liste
git lfs ls-files --size         # Avec taille
git lfs ls-files | wc -l        # Compte
```

### Vérifier qu'un fichier est bien en LFS
```bash
git check-attr --all data/dicom/raw/some.dcm
# Doit afficher : filter: lfs, diff: lfs, merge: lfs
```

---

## 📤 Ajouter de nouveaux gros fichiers

### Cas 1 : le type est déjà dans `.gitattributes`

Pour `*.dcm, *.zip, *.pth, *.ckpt, *.h5, *.onnx, *.safetensors, *.bin, *.pkl, *.joblib, data/**/*.png` : **rien à faire**, LFS catch automatiquement.

```bash
cp ~/Downloads/nouveau_modele.pth models/dicom/
git add models/dicom/nouveau_modele.pth
git commit -m "feat: ajout modèle DICOM v2"
git push
# → upload LFS automatique
```

### Cas 2 : nouveau type de fichier (ex: .parquet)

```bash
# 1. Activer LFS pour ce pattern
git lfs track "*.parquet"
# → met à jour .gitattributes

# 2. Commit .gitattributes
git add .gitattributes
git commit -m "chore: track .parquet via LFS"

# 3. Ajouter le fichier normalement
git add data/text_classification/embeddings.parquet
git commit -m "feat: ajout embeddings"
git push
```

### Cas 3 : convertir un fichier déjà commit en LFS

Si tu as accidentellement commit un gros fichier sans LFS :

```bash
# Sur ta branche (jamais sur main directement) :
git lfs migrate import --no-rewrite --include="*.parquet"
git push
```

Ne réécrit que ton dernier commit. **À éviter sur main** sans coordination équipe.

---

## 📁 Conventions de placement

### Où mettre quel type de fichier

| Type | Emplacement | LFS auto ? |
|---|---|---|
| `.dcm` (DICOM brut) | `data/dicom/raw/` | ✅ oui |
| `.zip` (datasets) | `data/<module>/` | ✅ oui |
| `.png` preprocessed (>1 Mo typique) | `data/**/` | ✅ oui (sous data/) |
| `.png` figures, plots | `docs/figures/` | ❌ non (volontaire) |
| `.pth`, `.ckpt`, `.safetensors` | `models/<module>/` | ✅ oui |
| `.csv` petit (< 1 Mo) | `data/<module>/` | ❌ non |
| `.csv` lourd (> 50 Mo) | `data/<module>/` | ❌ non — préférer `.parquet` ou activer LFS |
| **Datasets partagés multi-modules** | `data/shared/` | selon ext |
| Caches dérivés (regenerable) | `data/cache/` | gitignored |

### Naming convention pour modèles

✅ Bon :
- `models/text_classification/cxr_bert_openi_auc0.91.pth`
- `models/dicom/densenet121_multiwindow_v2.ckpt`

❌ Mauvais :
- `models/model.pth`
- `models/v3_final_final.ckpt`

Inclure : **architecture + dataset + métrique principale** (ou version sémantique).

---

## 🚫 Ce qu'il NE faut JAMAIS commit

- ❌ Environnements virtuels (`venv/`, `.venv/`, `**/site-packages/`)
- ❌ Checkpoints intermédiaires d'entraînement (`epoch=*.ckpt`) → mets-les dans `**/lightning_logs/` (gitignored)
- ❌ Caches de notebooks (`.ipynb_checkpoints/`) — gitignored
- ❌ Secrets, clés SSH, tokens API → règles de sécurité dans `.gitignore`
- ❌ Données patient **non anonymisées** (DICOM avec PatientName) — risque RGPD
- ❌ Fichiers > 100 Mo qui ne sont **pas** des datasets structurels (artefacts temporaires, dumps de debug)

---

## 🩺 Problèmes fréquents

### "fatal: this exceeds GitHub's file size limit of 100 MB"
Tu push vers le remote `github`. Le miroir GitHub ne supporte pas les blobs LFS sur le compte gratuit. **Solution** : push uniquement vers `origin` (GitLab) :
```bash
git push origin <branche>
```

### "smudge filter lfs failed"
Le fichier LFS n'a pas pu être téléchargé. Vérifie :
```bash
git lfs fsck         # Vérifie l'intégrité
git lfs fetch        # Re-télécharge
git lfs checkout     # Re-applique les fichiers
```

### Le fichier sur disque fait 1 KB et contient juste du texte LFS
Tu n'as pas pull les blobs LFS. Lance `git lfs pull` (ou `git lfs pull --include="<path>"`).

### Quota LFS GitLab dépassé
Va sur https://gitlab.enst.fr → projet MM-MIA → `Settings` → `Usage Quotas`. Demander une extension au support DSI Télécom Paris si besoin.

### Tu as commit un gros fichier sans LFS et tu veux annuler
```bash
# Si pas encore push :
git reset --soft HEAD~1
git rm --cached <gros_fichier>
git lfs track "*.<ext>"   # si nouveau type
git add .gitattributes <gros_fichier>
git commit -m "fix: <gros_fichier> en LFS"

# Si déjà push : utiliser git lfs migrate import sur ta branche (voir Cas 3 ci-dessus)
```

---

## 🔄 Workflow complet — exemple type

```bash
# 1. Récupérer les dernières modifs
git checkout main
git pull
git lfs pull           # important : récupérer aussi les blobs

# 2. Créer une branche
git checkout -b feat/new-densenet-checkpoint

# 3. Travailler, entraîner un modèle, sauvegarder
# (entraînement local…)
cp lightning_logs/best.ckpt models/dicom/densenet121_v3_auc0.94.ckpt

# 4. Commit + push
git add models/dicom/densenet121_v3_auc0.94.ckpt models/README.md
git commit -m "feat(dicom): ajout DenseNet121 v3 (AUC=0.94)"
git push -u origin feat/new-densenet-checkpoint
# → blob LFS uploadé vers GitLab, pointeur dans le commit

# 5. Créer MR sur gitlab.enst.fr, review équipe, merge
```

---

## 📚 Pour aller plus loin

- [Doc officielle Git LFS](https://git-lfs.com/)
- [Cheatsheet LFS Atlassian](https://www.atlassian.com/git/tutorials/git-lfs)
- Notre `.gitattributes` racine : règles actuelles
- Notre `.lfsconfig` : pourquoi LFS pointe vers GitLab uniquement
