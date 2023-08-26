# Breast Cancer Prediction

This repository contains the implementation of a breast cancer prediction model using machine learning techniques. The code is written in Python and utilizes libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Keras.

## Dataset

The breast cancer dataset used in this project contains information about various features related to breast cancer diagnosis. It includes characteristics of cell nuclei present in the breast biopsies. The target variable is binary, indicating the diagnosis as either malignant (M) or benign (B). 

## Methodology

The breast cancer prediction model follows a comprehensive methodology that encompasses data preprocessing, model construction, hyperparameter tuning, training, evaluation, and result visualization:

1. **Data Loading and Exploration:** The notebook begins by loading the breast cancer dataset, which contains a range of features characterizing cell nuclei properties. The dataset is explored using Pandas, providing insights into the data's structure and characteristics.

2. **Data Preprocessing:** Categorical data is encoded using the LabelEncoder from Scikit-learn to convert the 'diagnosis' column from binary labels ('M' for malignant and 'B' for benign) to numeric values (1 and 0). The dataset is also normalized using Min-Max scaling to bring all features within a consistent range.

3. **Train-Test Split:** The dataset is split into training and test sets using Scikit-learn's `train_test_split` function. This enables the model to learn patterns from the training data and evaluate its performance on unseen test data.

4. **Neural Network Architecture:** A neural network model is constructed using Keras, a high-level neural networks API. The model architecture includes two hidden layers with ReLU activation functions and an output layer with a sigmoid activation function for binary classification. The model is compiled with the Adam optimizer and binary cross-entropy loss.

5. **Hyperparameter Tuning:** GridSearchCV from Scikit-learn is employed to fine-tune hyperparameters such as batch size and the number of training epochs. This technique explores multiple combinations to identify the optimal hyperparameters for the model.

6. **Model Training:** The neural network model is trained on the training data using Keras' `fit` function. The training process is monitored, and accuracy and loss metrics are recorded for both training and validation sets.

7. **Result Visualization:** The notebook visualizes the training and validation accuracy/loss trends over epochs using Matplotlib. These visualizations provide insights into the model's learning progress and potential overfitting.

8. **Model Evaluation:** The model's predictive performance is evaluated using the test dataset. Predictions are compared against the ground truth labels, and a confusion matrix is generated. The confusion matrix helps in understanding the model's classification performance.

9. **Confusion Matrix Visualization:** A Seaborn heatmap is used to visualize the confusion matrix, highlighting the true positive, true negative, false positive, and false negative predictions.

10. **Accuracy Calculation:** The accuracy score is calculated based on the confusion matrix, providing a percentage that represents the model's overall classification accuracy on the test data.

## Usage

1. Clone the repository.

2. Navigate to the repository directory.

3. Run the provided Jupyter Notebook named `Breast Cancer Prediction.ipynb` in a compatible environment like Google Colab or Jupyter Notebook.

4. The notebook contains detailed explanations of each step. Execute the code cells sequentially to observe the methodology in action.

## Notes

1. Hyperparameter tuning is demonstrated using GridSearchCV to find the optimal batch size and number of epochs for the neural network.
2. Adjusting model architecture and hyperparameters can lead to different results and performance improvements.

