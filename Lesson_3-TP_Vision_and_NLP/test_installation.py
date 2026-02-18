#!/usr/bin/env python3
"""
Script de test rapide pour vérifier que les notebooks du Lesson 3 peuvent s'exécuter.
À exécuter avant le cours pour vérifier l'installation.
"""

import sys

def test_imports():
    """Vérifier que tous les packages nécessaires sont installés."""
    print("🔍 Vérification des imports...")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers (HF)"),
        ("datasets", "Datasets (HF)"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("requests", "Requests"),
    ]
    
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            failed.append(name)
    
    return failed

def test_torch():
    """Vérifier que PyTorch fonctionne."""
    print("\n🔥 Test PyTorch...")
    import torch
    
    # Créer un tenseur simple
    x = torch.randn(3, 3)
    print(f"  ✅ Tenseur créé : {x.shape}")
    
    # Vérifier le device
    if torch.cuda.is_available():
        print(f"  ✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ℹ️  CPU uniquement (pas de CUDA)")
    
    # Test opération
    y = x @ x.T
    print(f"  ✅ Opération matmul fonctionne : {y.shape}")

def test_transformers():
    """Vérifier que Transformers fonctionne."""
    print("\n🤗 Test HuggingFace Transformers...")
    from transformers import AutoTokenizer, AutoModel
    
    try:
        # Charger un modèle tiny pour test rapide
        model_name = "prajjwal1/bert-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test inférence
        inputs = tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  ✅ Modèle chargé : {model_name}")
        print(f"  ✅ Inférence fonctionne : {outputs.last_hidden_state.shape}")
    except Exception as e:
        print(f"  ⚠️  Impossible de charger le modèle (connexion Internet?) : {e}")

def test_vision():
    """Vérifier que torchvision fonctionne."""
    print("\n🖼️  Test TorchVision...")
    import torchvision
    from torchvision import models
    
    # Charger ResNet18
    model = models.resnet18(pretrained=False)
    print(f"  ✅ ResNet18 chargé : {sum(p.numel() for p in model.parameters()):,} params")

def main():
    print("=" * 50)
    print("Test d'installation - Lesson 3")
    print("=" * 50)
    
    failed = test_imports()
    
    if failed:
        print(f"\n❌ Packages manquants : {', '.join(failed)}")
        print("\nInstallez-les avec :")
        print("  pip install " + " ".join(failed))
        sys.exit(1)
    
    test_torch()
    test_transformers()
    test_vision()
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests sont passés !")
    print("Les notebooks du Lesson 3 sont prêts à être utilisés.")
    print("=" * 50)

if __name__ == "__main__":
    main()
