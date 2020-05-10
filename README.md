# Emotion_detection
It is a project for emotion detection with EEG and ECG signals.
Different signals are processed, including feature extraction and normalization, in `EEG.py` and `ECG.py` respectively and saved as `EEG.csv` and `ECG.csv`.
After that features are fused in `Datafusion.py` and saved as `feature.csv`.
With grid searching, proper model's parameters are selected in `model.py`.
The data is aquired from the `DREAMER A Database for Emotion RecognitionThrough EEG and ECG Signals From Wireless Low-cost Off the-Shelf Devices note.pdf`.
