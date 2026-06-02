# Mémoire Master en Sciences Informatiques

Voici une documentation exhaustive de la manière d'utiliser ce repos

## Téléchargement des checkpoints

### Avant de commencer

1. Télécharger l'outil CLI de Huggingface :

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

2. Se connecter à Huggingface avec vos identifiants :

```bash
hf auth login
```

### Sam3

1. Se déplacer dans le dossier du checkpoint de SAM3 :

```bash
cd  code/checkpoints/sam3/
```

2. Faire une demande d'accès au modèle via HuggingFace : https://huggingface.co/facebook/sam3

3. Télécharger le checkpoint de SAM3 :

```bash
hf download facebook/sam3 sam3.pt --local-dir .
```

4. Télécharger le vocabulaire utilisé par le modèle:

```bash
wget https://huggingface.co/spaces/LanguageBind/LanguageBind/resolve/main/open_clip/bpe_simple_vocab_16e6.txt.gz
```

### MoGe 2

1. Se déplacer dans le dossier du checkpoint de MoGe :

```bash
cd  code/checkpoints/moge/
```

2. Télécharger le checkpoint de MoGe :

```bash
hf download Ruicheng/moge-2-vitl model.pt --local-dir .
```

### TACO

1. Se déplacer dans le dossier du checkpoint de TACO :

```bash
cd  code/checkpoints/taco/
```

2. Télécharger le checkpoint de TACO :

```bash
wget https://huggingface.co/datasets/JasonAplp/TACO/resolve/main/checkpoints/last.ckpt
```

### TACO-Lora

1. Se déplacer dans le dossier du checkpoint de TACO :

```bash
cd  code/checkpoints/taco/
```

2. Faire une demande d'accès au modèle via HuggingFace : https://huggingface.co/Nicolasdelp/TACO-LoRA-Powerlifting

3. Télécharger le checkpoint de TACO LoRA :

```bash
hf download Nicolasdelp/TACO-LoRA-Powerlifting lora.ckpt --local-dir .
```

### Sam 3D Body

1. Se déplacer dans le dossier du checkpoint de SAM 3D Body :

```bash
cd  code/checkpoints/sam_3d_body/
```

2. Faire une demande d'accès au modèle via HuggingFace : https://huggingface.co/facebook/sam-3d-body-vith

3. Télécharger le checkpoint de SAM 3D Body :

```bash
hf download facebook/sam-3d-body-vith model.ckpt --local-dir .
```

4. Télécharger le checkpoint de MHR :

```bash
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
```

5. Extraire le checkpoint de MHR :

```bash
unzip -p assets.zip assets/mhr_model.pt > mhr_model.pt
```

6. Supprimer le zip:

```bash
rm assets.zip
```

## Création du dataset

### Avant de commencer

1. Installer Pixi :

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

2. Mettre à jour Pixi :

```bash
pixi self-update
```

### Téléchargement

1. Se déplacer dans le dossier data :

```bash
cd data/
```

2. Télécharger le dataset (dataset-tfe.tar, dataset-tfe-2.tar, dataset-tfe-3.tar) depuis l'emplacement dédié à l'Université de Mons

3. Extraire les différentes archives du dossier outputs :

```bash
tar -xf dataset-tfe.tar
tar -xf dataset-tfe-2.tar
tar -xf dataset-tfe-3.tar
```

```bash
for tar_file in outputs/IMG_*.tar; do
    tar -xf "$tar_file" -C outputs/
done
```

4. Lancer la création du dataset :

```bash
for video in data/inputs/IMG_00{01..28}.mov; do
    pixi run create-dataset-pipeline "$(basename "$video")";
done
```

5. Puis lancer cette commande pour créer les partitions train/val :

```bash
pixi run prepare-taco-dataset
```

## Fine-tuning de TACO

1. Créer un environement virtuel avec conda :

```bash
cd code/models/taco/
conda env create -f environment.yml
conda activate taco
```

2. Lancer fine-tuning :

```bash
cd code/models/taco/
conda activate taco
bash train_continue.sh
```

## Utilisation du code

### Pipeline principal

#### Lancer le pipeline

```bash
pixi run main-pipeline
```

(Pour choisir une vidéo spécifique, se rendre dans le pixi.toml)

### Application web

#### Télécharger les représentations 3D du rapport

1. Se déplacer dans le dossier data :

```bash
cd code/web/public/
```

2. Télécharger le dataset (dataset-web.tar) depuis l'emplacement dédié à l'Université de Mons

3. Extraire les différentes archives du dossier public :

```bash
tar -xf dataset-web.tar
```

4. Télécharger et installer Node.js

Rendez-vous sur cette page pour la marche à suivre : https://nodejs.org/en/download

5. Lancer l'application web

```bash
cd code/web/
npm run dev
```

Pour choisir une vidéo spécifique, se rendre dans /app/page.tsx et changer le `type` (squat, bench ou deadlift)

#### Transférer des nouvelles représentations 3D (après le traitement via le pipeline principal)

```bash
for video in IMG_XXXX IMG_XXXX IMG_XXXX; do
    mkdir -p /media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/code/web/public/$video
    find /media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/outputs/$video/original/sam3dbody/lora/ \
      -name "mesh_predicted_frame_*_mesh_000.ply" -o \
      -name "mesh_predicted_frame_*_keypoints.json" | \
      xargs -I {} cp {} /media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/code/web/public/$video/
done
```
