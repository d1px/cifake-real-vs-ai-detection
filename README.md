# CIFAKE - Real vs AI Image Detector

My computer vision project for CMS22202 at Ravensbourne University London.
Level 5 BSc Computer Science.

This project uses a CNN to detect if an image is real or AI generated.
I trained two models - one for general images (CIFAKE dataset) and one for faces.
There is also a live webcam demo with a pygame UI.

## How to run

```bash
python setup.py
python start.py
```

To run the webcam demo while on a video call (so both apps can use the camera):

```bash
bash setup_virtual_camera.sh
bash go.sh
```

## Files

- `train.py` - trains the main cifake model
- `train_faces.py` - trains the face detection model
- `camera.py` - live webcam demo
- `predict.py` - predict a single image
- `test_realworld.py` - test a folder of images
- `retrain_robust.py` - fine tuning script with more augmentation
- `start.py` - launches the demo
- `setup.py` - installs dependencies
- `go.sh` - starts virtual camera then launches detector
- `launch_demo.sh` - alternative launcher
- `setup_virtual_camera.sh` - sets up /dev/video10 with v4l2loopback

## Models

- `cifake_model.h5` - trained on CIFAKE dataset, 92.4% accuracy
- `cifake_faces_model.h5` - trained on 140k faces dataset, ~93% accuracy

## Datasets

Downloaded from Kaggle:
- CIFAKE dataset (real and AI generated images)
- 140k Real and Fake Faces dataset

## What I used

Python, TensorFlow, Keras, OpenCV, pygame, NumPy, scikit-learn
