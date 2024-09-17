## Tomato Disease Detection Model - VGG19

This project implements a **Tomato Disease Detection Model** using the **VGG19** deep learning architecture. The goal of this model is to classify and diagnose tomato leaf diseases based on images provided by farmers. The model was trained on a dataset of various tomato leaf images that belong to different disease categories, aiming to assist farmers in identifying potential crop issues at an early stage, helping them take preventive actions to minimize losses.

### Features:
- **Deep Learning Architecture**: The model leverages the power of the **VGG19** convolutional neural network, which is pre-trained on ImageNet and fine-tuned on the tomato leaf disease dataset.
- **Image Classification**: The model categorizes images of tomato leaves into healthy or one of the various disease classes.
- **Accuracy**: Achieved an accuracy of **84.67%** on the validation dataset after fine-tuning, making it effective for real-world agricultural scenarios.
- **Model Training and Evaluation**: The notebook contains code to load the dataset, preprocess the images, and train the VGG19 model. After training, evaluation metrics such as accuracy, precision, and recall are computed to assess the model's performance.
- **Deployment**: The model is deployed using **Flask** for easy accessibility, allowing users to upload images and get predictions directly through a web interface.

### Usage:
1. Clone the repository.
2. Install the required dependencies from the `requirements.txt` file.
3. Run the Jupyter Notebook `vgg19_tomato_disease_detection.ipynb` to train and evaluate the model.
4. Use the provided Flask application to deploy the model for real-time inference.

### Dataset:
The model is trained on a dataset of tomato leaves that includes multiple disease classes and healthy leaf images. The data is augmented to improve model robustness and generalization.

