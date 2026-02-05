#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de génération d'illustrations pour les slides du cours NLP
Illustrations sur le pré-entraînement et le fine-tuning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors
import seaborn as sns

# Créer le dossier d'images s'il n'existe pas
os.makedirs('images/generated', exist_ok=True)

# Configuration des styles pour les plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_pretraining_vs_finetuning():
    """
    Crée une illustration montrant la différence entre pré-entraînement et fine-tuning
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Définition des couleurs
    pretraining_color = "#6495ED"  # Bleu clair
    finetuning_color = "#FF7F50"   # Corail
    
    # Paramètres de position
    pretraining_y = 6
    finetuning_y = 2
    x_start = 1
    x_end = 9
    
    # Dessiner la ligne de pré-entraînement
    ax.plot([x_start, x_end], [pretraining_y, pretraining_y], 
            color=pretraining_color, linewidth=12, alpha=0.6)
    
    # Dessiner la ligne de fine-tuning
    ax.plot([7, 9], [finetuning_y, finetuning_y], 
            color=finetuning_color, linewidth=12, alpha=0.6)
    
    # Dessiner la flèche du pré-entraînement au fine-tuning
    arrow = FancyArrowPatch((6.5, pretraining_y), (6.5, finetuning_y), 
                           connectionstyle="arc3,rad=0", 
                           arrowstyle="fancy,head_length=10,head_width=10", 
                           color="black", linewidth=2)
    ax.add_patch(arrow)
    
    # Ajouter des icônes pour représenter les données
    # Pré-entraînement: grande quantité de données générales
    for i in range(x_start, 7):
        ax.text(i, pretraining_y + 0.8, "📄", fontsize=20)
    
    # Fine-tuning: petite quantité de données spécifiques
    ax.text(7.5, finetuning_y + 0.8, "📄", fontsize=20, color="darkred")
    ax.text(8.0, finetuning_y + 0.8, "📄", fontsize=20, color="darkred")
    ax.text(8.5, finetuning_y + 0.8, "📄", fontsize=20, color="darkred")
    
    # Ajouter des étiquettes et des annotations
    ax.text(3, pretraining_y - 0.8, "Pré-entraînement", fontsize=18, 
            color=pretraining_color, fontweight='bold', ha='center')
    ax.text(3, pretraining_y - 1.4, "Données massives et générales\nObjectif: apprendre des représentations générales", 
            fontsize=14, color="gray", ha='center')
    
    ax.text(8, finetuning_y - 0.8, "Fine-tuning", fontsize=18, 
            color=finetuning_color, fontweight='bold', ha='center')
    ax.text(8, finetuning_y - 1.4, "Données spécifiques\nObjectif: adaptation à une tâche", 
            fontsize=14, color="gray", ha='center')
    
    # Ajouter un modèle
    model_x = 6.5
    model_y = 4
    model_rect = Rectangle((model_x-1, model_y-1), 2, 2, 
                           facecolor="lightgray", edgecolor="black", alpha=0.7)
    ax.add_patch(model_rect)
    ax.text(model_x, model_y, "Modèle", fontsize=16, ha='center', va='center')
    
    # Configurer les axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Sauvegarder l'image
    plt.tight_layout()
    plt.savefig('images/generated/pretraining_vs_finetuning.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_vs_partial_finetuning():
    """
    Crée une illustration montrant la différence entre le full fine-tuning et le partial fine-tuning
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Définition des couleurs
    frozen_color = "#B0C4DE"  # Bleu clair grisé
    trainable_color = "#FF7F50"  # Corail
    adapter_color = "#9370DB"  # Violet moyen
    
    # Structure du modèle
    num_layers = 6
    layer_height = 0.8
    model_width = 4
    spacing = 0.3
    
    # Positions des deux modèles
    full_x = 3
    partial_x = 11
    
    # Full Fine-tuning
    ax.text(full_x, 8, "Full Fine-tuning", fontsize=20, fontweight='bold', ha='center')
    
    for i in range(num_layers):
        y_pos = 6 - i * (layer_height + spacing)
        rect = Rectangle((full_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=trainable_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(full_x, y_pos + layer_height/2, f"Couche {i+1}", 
                ha='center', va='center', fontsize=12, color='black')
    
    # Partial Fine-tuning
    ax.text(partial_x, 8, "Partial Fine-tuning", fontsize=20, fontweight='bold', ha='center')
    
    # Layers figés
    for i in range(num_layers-2):
        y_pos = 6 - i * (layer_height + spacing)
        rect = Rectangle((partial_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=frozen_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(partial_x, y_pos + layer_height/2, f"Couche {i+1} (figée)", 
                ha='center', va='center', fontsize=12, color='black')
    
    # Layers entraînables
    for i in range(num_layers-2, num_layers):
        y_pos = 6 - i * (layer_height + spacing)
        rect = Rectangle((partial_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=trainable_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(partial_x, y_pos + layer_height/2, f"Couche {i+1} (entraînable)", 
                ha='center', va='center', fontsize=12, color='black')
    
    # Légende
    legend_y = 0.8
    
    # Full fine-tuning
    rect1 = Rectangle((full_x - 2, legend_y), 0.5, 0.5, 
                     facecolor=trainable_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect1)
    ax.text(full_x - 1.2, legend_y + 0.25, "Paramètres entraînables", 
            va='center', fontsize=12)
    
    # Partial fine-tuning
    rect2 = Rectangle((partial_x - 2, legend_y), 0.5, 0.5, 
                     facecolor=frozen_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect2)
    ax.text(partial_x - 1.2, legend_y + 0.25, "Paramètres figés", 
            va='center', fontsize=12)
    
    # Configurer les axes
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Sauvegarder l'image
    plt.tight_layout()
    plt.savefig('images/generated/full_vs_partial_finetuning.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_finetuning_methods():
    """
    Crée une illustration montrant différentes méthodes de fine-tuning (adapter, LoRA, prompt-tuning)
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Définition des couleurs
    frozen_color = "#B0C4DE"  # Bleu clair grisé
    trainable_color = "#FF7F50"  # Corail
    adapter_color = "#9370DB"  # Violet moyen
    lora_color = "#20B2AA"  # Turquoise clair
    prompt_color = "#FFD700"  # Or
    
    # Structure du modèle
    num_layers = 5
    layer_height = 0.8
    model_width = 3
    spacing = 0.3
    
    # Positions des trois modèles
    adapter_x = 3
    lora_x = 8
    prompt_x = 13
    
    # Titre
    ax.text(8, 9.5, "Méthodes avancées de fine-tuning", fontsize=22, fontweight='bold', ha='center')
    
    # Adapter Fine-tuning
    ax.text(adapter_x, 8.7, "Adapter Fine-tuning", fontsize=18, fontweight='bold', ha='center')
    
    for i in range(num_layers):
        y_pos = 7 - i * (layer_height + spacing)
        # Couche principale figée
        rect = Rectangle((adapter_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=frozen_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Adapter module
        if i > 0 and i < num_layers-1:  # Ajouter des adapters aux couches intermédiaires
            adapter_width = model_width * 0.4
            adapter_height = layer_height * 0.6
            adapter = Rectangle((adapter_x + model_width/2, y_pos + layer_height/2 - adapter_height/2), 
                               adapter_width, adapter_height, 
                               facecolor=adapter_color, edgecolor='black', alpha=0.9)
            ax.add_patch(adapter)
            ax.text(adapter_x + model_width/2 + adapter_width/2, y_pos + layer_height/2, "Adapter", 
                   ha='center', va='center', fontsize=10, color='white')
            
            # Flèches
            arrow1 = FancyArrowPatch((adapter_x + model_width/2, y_pos + layer_height/2), 
                                    (adapter_x + model_width/2 + 0.1, y_pos + layer_height/2),
                                    arrowstyle='->', mutation_scale=10)
            ax.add_patch(arrow1)
            
            arrow2 = FancyArrowPatch((adapter_x + model_width/2 + adapter_width, y_pos + layer_height/2), 
                                    (adapter_x + model_width/2 + adapter_width + 0.1, y_pos + layer_height/2),
                                    arrowstyle='->', mutation_scale=10)
            ax.add_patch(arrow2)
        
        ax.text(adapter_x, y_pos + layer_height/2, f"Couche {i+1}", 
                ha='center', va='center', fontsize=12, color='black')
    
    # LoRA Fine-tuning
    ax.text(lora_x, 8.7, "LoRA Fine-tuning", fontsize=18, fontweight='bold', ha='center')
    
    for i in range(num_layers):
        y_pos = 7 - i * (layer_height + spacing)
        # Couche principale figée
        rect = Rectangle((lora_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=frozen_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Modules LoRA
        if i > 0 and i < num_layers-1:  # Ajouter LoRA aux couches intermédiaires
            lora_height = layer_height * 0.3
            
            # Low-rank matrix A
            lora_a = Rectangle((lora_x - model_width/4, y_pos - lora_height - 0.1), 
                              model_width/2, lora_height, 
                              facecolor=lora_color, edgecolor='black', alpha=0.9)
            ax.add_patch(lora_a)
            ax.text(lora_x, y_pos - lora_height - 0.1 + lora_height/2, "A (rang faible)", 
                   ha='center', va='center', fontsize=10, color='black')
            
            # Low-rank matrix B
            lora_b = Rectangle((lora_x - model_width/4, y_pos + layer_height + 0.1), 
                              model_width/2, lora_height, 
                              facecolor=lora_color, edgecolor='black', alpha=0.9)
            ax.add_patch(lora_b)
            ax.text(lora_x, y_pos + layer_height + 0.1 + lora_height/2, "B (rang faible)", 
                   ha='center', va='center', fontsize=10, color='black')
            
            # Flèches
            arrow_a = FancyArrowPatch((lora_x, y_pos - 0.1), 
                                    (lora_x, y_pos - lora_height - 0.1),
                                    arrowstyle='->', mutation_scale=10)
            ax.add_patch(arrow_a)
            
            arrow_b = FancyArrowPatch((lora_x, y_pos + layer_height + 0.1 + lora_height), 
                                    (lora_x, y_pos + layer_height + 0.1),
                                    arrowstyle='->', mutation_scale=10)
            ax.add_patch(arrow_b)
            
            # Flèches de retour
            arrow_back = FancyArrowPatch((lora_x + model_width/4, y_pos - lora_height/2), 
                                       (lora_x + model_width/4 + 0.3, y_pos + layer_height/2),
                                       connectionstyle="arc3,rad=0.3",
                                       arrowstyle='->', mutation_scale=10)
            ax.add_patch(arrow_back)
        
        ax.text(lora_x, y_pos + layer_height/2, f"Couche {i+1}", 
                ha='center', va='center', fontsize=12, color='black')
    
    # Prompt Tuning
    ax.text(prompt_x, 8.7, "Prompt Tuning", fontsize=18, fontweight='bold', ha='center')
    
    for i in range(num_layers):
        y_pos = 7 - i * (layer_height + spacing)
        # Couche principale figée
        rect = Rectangle((prompt_x - model_width/2, y_pos), model_width, layer_height, 
                         facecolor=frozen_color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(prompt_x, y_pos + layer_height/2, f"Couche {i+1}", 
                ha='center', va='center', fontsize=12, color='black')
    
    # Prompt tokens (seulement en entrée)
    prompt_width = model_width * 0.7
    prompt_height = layer_height
    prompt_tokens = Rectangle((prompt_x - model_width/2 - prompt_width - 0.3, 7), 
                             prompt_width, prompt_height, 
                             facecolor=prompt_color, edgecolor='black', alpha=0.9)
    ax.add_patch(prompt_tokens)
    ax.text(prompt_x - model_width/2 - prompt_width/2 - 0.3, 7 + prompt_height/2, 
            "Tokens de prompt\nentraînables", 
            ha='center', va='center', fontsize=10, color='black')
    
    # Flèche du prompt aux tokens d'entrée
    arrow_prompt = FancyArrowPatch((prompt_x - model_width/2 - 0.3, 7 + layer_height/2), 
                                 (prompt_x - model_width/2, 7 + layer_height/2),
                                 arrowstyle='->', mutation_scale=10)
    ax.add_patch(arrow_prompt)
    
    # Légende
    legend_y = 1.5
    legend_x1 = 3
    legend_x2 = 8
    legend_x3 = 13
    
    # Paramètres figés
    rect1 = Rectangle((legend_x1 - 2, legend_y), 0.4, 0.4, 
                     facecolor=frozen_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect1)
    ax.text(legend_x1 - 1.5, legend_y + 0.2, "Paramètres figés", 
            va='center', fontsize=12)
    
    # Adapter
    rect2 = Rectangle((legend_x1 - 2, legend_y - 0.6), 0.4, 0.4, 
                     facecolor=adapter_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect2)
    ax.text(legend_x1 - 1.5, legend_y - 0.4, "Modules adapter", 
            va='center', fontsize=12)
    
    # LoRA
    rect3 = Rectangle((legend_x2 - 2, legend_y), 0.4, 0.4, 
                     facecolor=lora_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect3)
    ax.text(legend_x2 - 1.5, legend_y + 0.2, "Matrices de rang faible (LoRA)", 
            va='center', fontsize=12)
    
    # Prompt tuning
    rect4 = Rectangle((legend_x3 - 2, legend_y), 0.4, 0.4, 
                     facecolor=prompt_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect4)
    ax.text(legend_x3 - 1.5, legend_y + 0.2, "Tokens de prompt", 
            va='center', fontsize=12)
    
    # Caractéristiques
    stats_y = 0.5
    ax.text(legend_x1, stats_y, "Adapter:\n• ~1-2% des paramètres\n• Petit module entre couches\n• Préserve performances", 
            va='top', fontsize=11)
    
    ax.text(legend_x2, stats_y, "LoRA:\n• < 1% des paramètres\n• Matrices de rang faible\n• Plus efficace en mémoire", 
            va='top', fontsize=11)
    
    ax.text(legend_x3, stats_y, "Prompt Tuning:\n• < 0.1% des paramètres\n• Seulement tokens d'entrée\n• Compatible inférence", 
            va='top', fontsize=11)
    
    # Configurer les axes
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Sauvegarder l'image
    plt.tight_layout()
    plt.savefig('images/generated/finetuning_methods.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_pretraining_vs_finetuning()
    plot_full_vs_partial_finetuning()
    plot_finetuning_methods()
    print("Images générées avec succès dans le dossier 'images/generated/'") 