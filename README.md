# CIFAKE: Real vs AI-Generated Image Detection

**Ravensbourne University London | CMS22202 Computer Vision and AI | Level 5 BSc Computer Science**

## Project Overview
A CNN-based binary image classifier that detects whether an image is a real photograph or AI-generated. Built using TensorFlow and Keras with a live real-time webcam demo.

## Models
- `cifake_model.h5` — CIFAKE object classifier (92.4% accuracy)
- `cifake_faces_model.h5` — Face classifier (93% accuracy after fine-tuning)

## Datasets
- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Files
- `train.py` — Train CIFAKE object model
- `train_faces.py` — Train faces model
- `camera.py` — Live webcam demo with cyberpunk UI
- `predict.py` — Classify a single image
- `test_realworld.py` — Batch test real world images
- `start.py` — Launch the webcam demo
- `setup.py` — Install dependencies

## How to Run
```bash
python setup.py
python start.py
```

## Tech Stack
Python, TensorFlow, Keras, OpenCV, pygame, NumPy, scikit-learn
