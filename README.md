# Audio Event Detection Project

This is a student-friendly hackathon project for **Audio Event Detection**.
The goal is to take an audio clip and predict what sound is present in it, like dog bark, rain, siren, glass breaking, baby crying, and more.

## Why I changed the earlier version
The earlier version used TensorFlow CNN training, which can fail on many student laptops because of heavy installation and version issues. This version uses **Librosa + Scikit-learn**, so it is much easier to run locally and in Colab.

## Problem Statement
Build a model that can identify sound events from audio clips.

## Dataset
This project uses **ESC-50**.

Expected folder structure:

```text
audio_event_detection_hackathon_fixed/
├── ESC-50-master/
│   ├── audio/
│   └── meta/
│       └── esc50.csv
```

## Tech Stack
- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

## Project Structure
```text
audio_event_detection_hackathon_fixed/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
├── docs/
│   └── demo_video_script.md
├── notebooks/
│   └── Audio_Event_Detection_Colab.ipynb
└── src/
    ├── config.py
    ├── data_utils.py
    ├── features.py
    └── __init__.py
```

## Setup
```bash
pip install -r requirements.txt
```

Then download ESC-50 and keep the folder name as `ESC-50-master` in the project folder.

## Train the model
```bash
python train.py
```

This will create:

```text
artifacts/
├── audio_event_model.joblib
├── label_encoder.joblib
├── metrics.json
└── confusion_matrix.png
```

## Predict on one file
```bash
python predict.py ESC-50-master/audio/1-100032-A-0.wav
```

## Run the demo app
```bash
streamlit run app.py
```

## How this project works
1. Read audio files from ESC-50
2. Extract features using Librosa
3. Train a Random Forest classifier
4. Save model and label encoder
5. Predict sound event for a new file

## Why this version is better for a student
- easier to install
- works on CPU
- no TensorFlow dependency
- easier to explain in viva/demo
- faster to train than a CNN

## Limitations
- not real-time
- accuracy is lower than advanced deep learning models
- works best on clean short clips

## Future improvements
- add live microphone recording
- try data augmentation
- try CNN or transfer learning later
- deploy online

## Suggested Git commits
```bash
git init
git add .
git commit -m "Create project structure"
git commit -m "Add feature extraction and training code"
git commit -m "Add prediction script and Streamlit app"
git commit -m "Add notebook and documentation"
```

## Submission checklist
- public GitHub repo
- README.md
- source code
- Colab notebook link
- unlisted YouTube demo link
