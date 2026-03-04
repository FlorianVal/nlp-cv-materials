# Lesson 4 : Projet – Space Invaders par Vision par Ordinateur

Bienvenue dans le projet final ! L'objectif est de créer un **agent de contrôle basé sur la vision par ordinateur** pour jouer à Space Invaders — sans toucher au clavier.

## 🎯 Objectif

Le jeu tourne dans le navigateur. Votre script Python doit :
1. **Capturer** la webcam (ou l'écran) en temps réel
2. **Analyser** les images avec un modèle de vision
3. **Envoyer** les commandes au jeu via WebSocket

---

## 🗂 Structure

```
Lesson_4-Project/
└── space-invaders/          # Submodule – le jeu + serveur WebSocket
    ├── index.html           # Interface du jeu (ouvrir dans le navigateur)
    ├── server.js            # Serveur WebSocket (bridge Python ↔ jeu)
    ├── game.js              # Logique du jeu
    └── control_module.py    # Exemple de contrôle au clavier (référence)
```

---

## 🚀 Mise en route

### Prérequis

- Node.js (`node --version`)
- Python 3.8+ avec [uv](https://docs.astral.sh/uv/) (`brew install uv`)

### Lancer le jeu (3 terminaux)

**Terminal 1 – Serveur HTTP (jeu)**
```bash
cd space-invaders
python -m http.server 8000
```
Ouvrir [http://localhost:8000](http://localhost:8000) dans le navigateur.

**Terminal 2 – Serveur WebSocket (bridge)**
```bash
cd space-invaders
node server.js
```

**Terminal 3 – Module de contrôle**
```bash
# D'abord, tester avec le module clavier fourni :
cd space-invaders
uv run --with websockets python control_module.py

# Ensuite, lancer votre propre agent CV :
uv run --with websockets --with opencv-python --with mediapipe python ../cv_controller.py
```

---

## 🧩 Commandes disponibles

| Commande WebSocket | Effet         |
|--------------------|---------------|
| `LEFT`             | Déplacer à gauche |
| `RIGHT`            | Déplacer à droite |
| `FIRE`             | Tirer         |
| `ENTER`            | Valider / Démarrer |

---

## 💡 Sujet du projet

Créer `cv_controller.py` — un agent qui **utilise la caméra ou l'écran** pour décider des actions à envoyer au jeu.

### Approches suggérées (par ordre de difficulté)

#### Niveau 1 – Gestes mains (débutant)
Utiliser **MediaPipe Hands** pour détecter la position de la main et la mapper sur les commandes LEFT / RIGHT / FIRE.

```python
import mediapipe as mp

# Idée : main à gauche → LEFT, à droite → RIGHT, doigt levé → FIRE
```

#### Niveau 2 – Pose du corps (intermédiaire)
Utiliser **MediaPipe Pose** pour détecter l'inclinaison du corps ou les gestes bras.

#### Niveau 3 – Objet de couleur (intermédiaire)
Détecter un objet de couleur spécifique (balle, feutre…) avec OpenCV et l'utiliser comme joystick.

```python
import cv2
import numpy as np

# Seuillage HSV → trouver le centroïde → mapper sur LEFT/RIGHT
```

#### Niveau 4 – Capture d'écran + vision (avancé)
Capturer l'état du jeu directement depuis l'écran avec `mss` ou `pyautogui`, localiser le vaisseau et les ennemis, puis construire une logique de décision ou un agent entraîné par renforcement.

---

## 📦 Lancer votre script avec uv

Pas besoin de créer un environnement virtuel. `uv` gère tout automatiquement :

```bash
uv run --with websockets --with opencv-python --with mediapipe python cv_controller.py
```

Ou ajoutez les dépendances directement en haut de votre script avec un bloc inline :

```python
# /// script
# dependencies = ["websockets", "opencv-python", "mediapipe", "mss"]
# ///
```

Puis lancez simplement :

```bash
uv run cv_controller.py
```

---

## 📝 Structure recommandée pour votre script

```python
import asyncio
import websockets
import cv2

WS_URI = "ws://localhost:8765"

def process_frame(frame) -> str | None:
    """
    Analyser une image et retourner une commande ou None.

    Returns:
        "LEFT", "RIGHT", "FIRE", "ENTER" ou None
    """
    # TODO : votre logique de vision ici
    return None


async def cv_controller():
    cap = cv2.VideoCapture(0)

    async with websockets.connect(WS_URI) as ws:
        print("Connecté au jeu !")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            command = process_frame(frame)

            if command:
                await ws.send(command)
                print(f"Envoyé : {command}")

            # Afficher le flux (optionnel)
            cv2.imshow("CV Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(cv_controller())
```

---

## ✅ Critères d'évaluation

| Critère | Points |
|---------|--------|
| Le script se connecte et envoie des commandes valides | 4 |
| La logique CV détecte correctement les gestes/positions | 6 |
| Le jeu est réellement contrôlable (démo live) | 4 |
| Qualité du code (lisibilité, structure) | 3 |
| Originalité / niveau de difficulté de l'approche | 3 |
| **Total** | **20** |

---

## 🔗 Ressources utiles

- [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)
- [OpenCV HSV Color Tracking](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [websockets Python docs](https://websockets.readthedocs.io/)
- [mss – screen capture](https://python-mss.readthedocs.io/)
