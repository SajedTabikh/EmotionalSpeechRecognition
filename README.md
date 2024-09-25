# Speech Emotion Recognition using CNN

## Introduction
This project aims to build a Speech Emotion Recognition (SER) system using a Convolutional Neural Network (CNN). The system classifies emotions such as **neutral, happy, sad, angry, fear, disgust,** and **surprise** from speech audio files by extracting relevant features.

## Project Overview
The project involves:
- Preprocessing speech data from multiple datasets.
- Applying data augmentation techniques to improve model performance.
- Extracting key audio features for emotion classification.
- Training a CNN model to predict emotions from the processed audio data.
- Evaluating the model using accuracy metrics, confusion matrix, and classification reports.

## Datasets Used
The following datasets were utilized in this project:
1. **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.
2. **TESS**: Toronto Emotional Speech Set.
3. **SAVEE**: Surrey Audio-Visual Expressed Emotion Dataset.
4. **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset.

Each dataset provides audio samples labeled with different emotions, allowing for comprehensive training and testing of the model.

## Data Preprocessing
- Audio data was preprocessed by extracting file paths and emotion labels from the filenames.
- Each dataset's structure was standardized by combining emotion labels and file paths into a single DataFrame.
- Audio augmentation techniques like **adding noise, time-stretching, pitch shifting,** and **time shifting** were applied to create variations in the dataset.
  
## Feature Extraction
Key features were extracted from the audio data using:
- **Zero Crossing Rate (ZCR)**: Measures the rate at which the audio signal changes sign.
- **Root Mean Square Energy (RMSE)**: Represents the intensity or energy of the signal.
- **Mel-Frequency Cepstral Coefficients (MFCC)**: Captures essential spectral properties of the audio signal.

These features were used to train the CNN model for emotion recognition.

## CNN Model Architecture
The model was designed with a series of 1D Convolutional layers followed by pooling and dropout layers. The final output layer applies the **softmax** function to classify the audio into one of the seven emotion categories. Batch normalization was applied to stabilize and accelerate training.

The model was trained using the **Adam optimizer** and **categorical crossentropy** as the loss function.

## Training and Evaluation
The model was trained on an 80-20 split of the data:
- **80%** for training.
- **20%** for testing.

The model achieved a **validation accuracy of 98.42%**, showing excellent performance in emotion classification. The performance metrics, such as precision, recall, and F1-score, were calculated for each emotion category.

## Model Performance
The model's performance was evaluated using:
- **Confusion Matrix**: Visualizes the correct and incorrect classifications.
- **Classification Report**: Provides precision, recall, and F1-score for each emotion.
  
The evaluation metrics showed high accuracy across all emotion categories.

## Data Augmentation Techniques
To ensure the model generalizes well to unseen data, the following augmentation techniques were applied:
1. **Noise Addition**: Simulates real-world environments with background noise.
2. **Time Stretching**: Adjusts the speed of the audio while maintaining pitch.
3. **Pitch Shifting**: Alters the pitch of the audio to simulate different speakers.
4. **Time Shifting**: Shifts the audio signal in time to mimic varying speech timings.

These techniques helped improve the robustness of the model.

## Saving and Deployment
The trained model, along with the scaler and encoder used for preprocessing, was saved for future use. The model can be easily loaded and used to predict emotions from new audio files.

## Future Work
Potential improvements for future iterations of the project include:
1. **Hyperparameter tuning** to further optimize the model.
2. Experimenting with **other deep learning architectures**, such as CNN-LSTM models, to capture both spatial and temporal features.
3. **Data augmentation** using more advanced techniques to increase dataset variability.

## Conclusion
The project successfully developed a Speech Emotion Recognition system with high accuracy. The use of data augmentation and feature extraction techniques significantly contributed to the model’s performance. The project showcases the potential of deep learning models in audio-based emotion detection.

## Requirements
To run this project, the following dependencies are required:
- Python 3.x
- TensorFlow
- Keras
- Librosa
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
