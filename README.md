ASL Landmark Classifier

A machine learning system for American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks and classical ML models.

The system extracts 21 hand landmarks (63 features) from a webcam and trains classifiers such as SVM, KNN, and Random Forest to recognize static ASL letters.

Features

MediaPipe hand landmark extraction

Webcam dataset collection

Classical ML training (SVM / KNN / RF)

Model comparison experiments

Confusion matrix visualization

Research experiment pipeline

Project Structure
dataset/
   raw/
      A/
      B/
      ...

dataset_builder.py
train_static.py
compare_models.py
evaluate.py
plot_confusion.py
utils_dataset.py

artifacts/
   trained models
   experiment results


Installation
pip install mediapipe opencv-python numpy scikit-learn matplotlib joblib
Build Dataset
python dataset_builder.py

This captures hand landmarks from the webcam and saves them as .npy feature vectors.

Train Model
python train_static.py --dataset_root dataset --read_folder raw --feature_mode raw --model svm
Compare Models
python compare_models.py --dataset_root dataset --read_folder raw
Generate Confusion Matrix
python plot_confusion.py --dataset_root dataset --read_folder raw --model_path artifacts/asl_svm_raw.joblib
Output

The project generates:

artifacts/
   asl_svm_raw.joblib
   results.json
   confusion_matrix.png

Author
Tonmoy Sarker
University of Wollongong — Computer Science (AI & Big Data)