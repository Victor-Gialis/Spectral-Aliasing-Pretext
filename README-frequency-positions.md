# SAP avec encodage positionnel en Hz

`--frequency_positions` donne au Transformer la fréquence centrale et la largeur de bande de chaque patch. Le mode conserve les fréquences d'échantillonnage natives.

## Préentraînement

Depuis la racine du dépôt, installez les dépendances avec `python -m pip install -r requirements.txt` et réutilisez votre `config/paths.py` pour indiquer où sont les données et les résultats.

```bash
python -m experiments.pretrain.pretrain_sap --pretrain_dataset PHM07 PHM09 PHM12 CWRU LASPI HUST PADERBORN --domain frequency --window_size 2048 --window_stride 256 --downsampling_factor 2 --symetric_spectrum --epochs 50 --frequency_positions
```

Vérifiez dans le journal que les sept datasets ont chargé. Utilisez `--epochs 1` pour un premier essai. Le mode Hz demande un facteur de sous-échantillonnage entier fixe et une fenêtre divisible par 32.

## Évaluation sur LASPI

```bash
python -m experiments.downstream.downstream_sap --pretrain_dataset Concat --downstream_dataset LASPI --backbone_checkpoint_path CHEMIN_DU_RUN --window_size 2048 --window_stride 2048 --data_ratio 0.1 --split_type speed_load_stratified --epochs 30 --head_type linear --task classification --seed 0
```

Remplacez `CHEMIN_DU_RUN` par le dossier qui contient `checkpoints/last.pt`. Le checkpoint active le mode Hz et fournit le facteur de repliement et la symétrisation. Utilisez `speed_stratified` sur CWRU et Paderborn. Le script downstream ne prend actuellement en charge ni HUST ni les trois PHM.

Pour comparer les deux encodages, relancez le préentraînement sans `--frequency_positions`. Côté downstream, vérifiez que le mode historique utilise le même facteur et la même symétrisation que le mode Hz, puis gardez le même split et le même budget de labels.
