# SMS-Spam-Classification-using-Machine-Learning
A machine learning project that classifies SMS messages as spam or ham (not spam) using deep learning techniques like LSTM. This repository includes the training and testing of the SMS Spam Collection dataset, preprocessing steps, model training, and a prediction function.
# SMS Spam Classification using Machine Learning

## Overview
This project aims to classify SMS messages as either **spam** or **ham** using deep learning. The model is trained on the **SMS Spam Collection Dataset** and utilizes an **LSTM** (Long Short-Term Memory) neural network to predict whether a message is spam or not. The project showcases preprocessing, training, and evaluation of a deep learning model for text classification.

## Project Structure
- `data/` - Contains the dataset files for training and testing.
- `model/` - Contains the trained model and script to predict new messages.
- `notebooks/` - Jupyter notebook for model training and testing.
- `scripts/` - Python scripts to preprocess data, train the model, and make predictions.

## Requirements
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- wget (for downloading dataset)

To install dependencies:

pip install -r requirements.txt
Dataset
The dataset used is the SMS Spam Collection Dataset, which consists of a collection of SMS messages labeled as spam (advertisements or unwanted messages) or ham (normal messages from friends). The dataset is available here.

How to Run
Clone the repository:


git clone https://github.com/yourusername/sms-spam-classification.git
cd sms-spam-classification
Install the required libraries:


pip install -r requirements.txt
Download the dataset using the provided script:

python download_data.py
Train the model:

python train_model.py
Make predictions:

python predict.py
Predicting SMS Messages
The predict_message function allows you to input an SMS and receive a prediction along with the confidence level of the message being spam or ham.

Example usage:


pred_text = "Win a free iPhone now!"
prediction = predict_message(pred_text)
print(prediction)  # Output: [0.85, 'spam']
Model Architecture
The model is an LSTM-based neural network that processes tokenized text and classifies it as either spam or ham.

Model Layers:
Embedding Layer: Converts words into dense vectors.

LSTM Layers: Capture sequential dependencies in the text.

Dense Layer: Output the prediction (spam or ham) using a sigmoid activation function.

Results
The model achieves an accuracy of over 98% on the test set after training for 5 epochs.

License
This project is licensed under the MIT License - see the LICENSE file for details.


---

### **Explanation of Sections:**

- **Title and Description**: Clearly explains the project’s purpose.
- **Project Structure**: Helps others navigate the directory.
- **Requirements**: Lists necessary Python libraries.
- **Dataset**: Mentions where the dataset comes from and includes a link for others to access.
- **How to Run**: Provides step-by-step instructions to set up and run the project.
- **Predicting SMS Messages**: Describes how users can predict messages after the model is trained.
- **Model Architecture**: Briefly explains the structure of the neural network.
- **Results**: Highlights the performance of the model.
