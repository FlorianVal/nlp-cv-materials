# Lesson 3 : TP Vision & NLP

Ce dossier contient les notebooks du TP pratique du cours 3.

## Structure

```
practicals/
├── 01_pytorch_huggingface_basics.ipynb    # Fondamentaux PyTorch & HF
└── 02_transfer_learning_vision.ipynb      # Transfer Learning en Vision
```

## Partie 1 : Fondamentaux PyTorch & HuggingFace

**Durée estimée :** 45-60 minutes

**Contenu :**
- Les tenseurs (images = tenseurs d'entiers !)
- La couche Linear (`y = x @ W^T + b`)
- Architecture d'un MLP simple
- Exploration de DistilBERT (couches, poids, shapes)
- Fine-tuning léger sur SST-2

**Points clés pédagogiques :**
- ✅ Fait le parallèle avec les cours précédents (CNN, Transformers)
- ✅ Montre explicitement ce qu'est une couche Linear
- ✅ Dive profond dans l'architecture d'un vrai modèle
- ✅ Utilise des modèles légers (~66M params) pour les PCs de fac

## Partie 2 : Transfer Learning en Vision

**Durée estimée :** 60-90 minutes

**Contenu :**
- Chargement de ResNet18 pré-entraîné
- Feature Extraction (geler le backbone)
- Fine-tuning complet
- Comparaison des stratégies
- Visualisation des prédictions et matrice de confusion
- Sauvegarde/chargement du modèle

**Points clés pédagogiques :**
- ✅ Modèle léger (ResNet18 ~11M params)
- ✅ Dataset réduit pour entraînement rapide sur CPU
- ✅ Compare explicitement les 2 stratégies
- ✅ Flux complet : data → train → eval → save

## Installation

Depuis ce dossier :

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -U pip
pip install -r requirements.txt
```

## Notes pour les étudiants

- Les notebooks sont conçus pour fonctionner sur **CPU** (pas de GPU nécessaire)
- Les temps d'entraînement sont courts (2-3 minutes par epoch)
- N'hésitez pas à modifier les hyperparamètres et observer l'impact !

## Exercices supplémentaires suggérés

1. Essayer d'autres architectures (MobileNetV2, EfficientNet-B0)
2. Tester sur d'autres datasets (Fashion-MNIST, CIFAR-100)
3. Implémenter un scheduler de learning rate
4. Ajouter du logging avec TensorBoard
