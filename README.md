# 🍚 **Rice Variety Classification** 🍚

Welcome to the **Rice Variety Classification** project! This project aims to classify different rice varieties based on their physical characteristics using machine learning techniques.

## 📖 **Project Overview**
This project utilizes image processing and machine learning algorithms to categorize rice grains based on features such as size, shape, and color. The goal is to provide an automated system for rice quality control and agricultural analysis.

## 🔧 **Technologies Used**
- Python
- TensorFlow / Keras
- split-folders

## **📊 Dataset Overview**
The dataset contains a total of 75,000 rice images across 5 rice varieties:

- Arborio
- Basmati
- Jasmine
- Ipsala
- Karacadag
  
The dataset was split into:

- 70% for training
- 15% for validation
- 15% for testing
The images were divided using the splitfolders module to ensure efficient and balanced training, validation, and testing sets.

## **Model Architecture**
The model was built using TensorFlow and includes the following layers:

1. Conv2D: Convolutional layers for feature extraction from images.
2. MaxPooling2D: Pooling layers to reduce the dimensionality of feature maps.
3. Flatten: Flattening the 2D data to feed into dense layers.
4. Dense: Fully connected layers for classification.
5. Dropout: Dropout layers to prevent overfitting and improve generalization.

## **📈 Results & Performance**
The model achieved a 99% validation accuracy, which indicates excellent performance in classifying the rice varieties. The high accuracy suggests that the model is effective in distinguishing between the different rice varieties based on their visual features.

| Image | Predicted Class | Confidence |
|-------|-----------------|------------|
| ![image1](https://github.com/adhi1702/Rice_Variety_Classification/blob/main/test/Arborio/Arborio%20(11206).jpg) | Arborio | 1.00 |
| ![image2](https://github.com/adhi1702/Rice_Variety_Classification/blob/main/test/Basmati/basmati%20(14788).jpg) | Basmati | 1.00 |
| ![image3](https://github.com/adhi1702/Rice_Variety_Classification/blob/main/test/Jasmine/Jasmine%20(14305).jpg) | Jasmine | 0.95 |
| ![image4](https://github.com/adhi1702/Rice_Variety_Classification/blob/main/test/Ipsala/Ipsala%20(12674).jpg) | Ipsala | 1.00 |
| ![image4](https://github.com/adhi1702/Rice_Variety_Classification/blob/main/test/Karacadag/Karacadag%20(13843).jpg) | Karacadag | 0.99 |

