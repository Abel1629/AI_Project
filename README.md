# AI_Project
Final project for the AI class

Project Goal:

The primary goal of this project is to build and evaluate a deep learning model that can classify chest X-ray images to detect whether a patient has Pneumonia or is NORMAL. It aims to automate or assist in the diagnostic process by providing a model that can distinguish between these two conditions based on visual features in the X-rays.

Key Components and Implementations:

Environment Setup:

The code begins by setting up the environment for execution in Google Colab, including mounting Google Drive to access the dataset.
It defines file paths for the training, validation, and test datasets, assuming they are organized in specific subdirectories within a 'chest_xray' folder.
Library Imports and Parameter Definition:

Essential libraries for deep learning (tensorflow, keras), data manipulation (numpy), plotting (matplotlib, seaborn), and utility functions (os, cv2) are imported.
Key parameters like the image size, batch size, random seed, and AUTOTUNE for data loading optimization are defined for reproducibility and efficiency.
Data Loading and Preprocessing:

The image_dataset_from_directory function is used to load the chest X-ray images directly from the specified directories, automatically inferring class labels from the directory structure.
The training directory is split into training and validation subsets.
The test dataset is loaded separately.
Images are resized to a consistent size (IMAGE_SIZE).
Pixel values are normalized to the range [0, 1] by dividing by 255.0.
The data loading pipeline is optimized using caching, shuffling (for training data), and prefetching (AUTOTUNE) for improved training speed.
Class names are extracted from the datasets.
Data Visualization:

A function is implemented to display a few sample images from the training dataset along with their corresponding class labels to visually inspect the data.
Model Definition (Transfer Learning with ResNet50):

Transfer learning is utilized by loading a pre-trained ResNet50 model (base_model) with weights from the ImageNet dataset. The top classification layer of ResNet50 is excluded.
The weights of the base_model are frozen (base_model.trainable = False) so they are not updated during the initial training phase, leveraging the learned features from ImageNet.
A new sequential model is built on top of the frozen base_model. This includes:
A GlobalAveragePooling2D layer to reduce the spatial dimensions of the feature maps.
A Dropout layer to help prevent overfitting.
A dense hidden layer with ReLU activation.
A final dense output layer with num_classes units and a softmax activation for multi-class probability prediction.
Model Compilation:

The model is compiled using the 'adam' optimizer, 'sparse_categorical_crossentropy' as the loss function (suitable for integer labels), and 'accuracy' as the evaluation metric.
Early Stopping Callback:

An EarlyStopping callback is defined to monitor the 'val_loss'.
Training will stop if the validation loss does not improve for a specified number of epochs (patience=10).
The model's weights will be restored to the point where the validation loss was lowest (restore_best_weights=True).
Model Training:

The model is trained using the model.fit() method on the training dataset (train_ds), with the validation dataset (val_ds) used for monitoring and early stopping. The maximum number of training epochs is set.
Model Evaluation:

The model's performance is evaluated on both the validation dataset (val_ds) and the separate test dataset (test_ds) to obtain the loss and accuracy metrics.
The results show a significant drop in performance from the validation set to the test set, indicating overfitting.
Training History Plot:

A function plot_history is implemented to visualize the training and validation accuracy and loss over the epochs. This plot helps diagnose overfitting by showing the divergence of the training and validation loss curves.
Detailed Evaluation on Test Set:

Predictions are made on the entire test dataset.
A classification_report is generated to provide detailed metrics (precision, recall, f1-score) for each class, highlighting the model's strengths and weaknesses.
A confusion_matrix is computed and visualized as a heatmap using seaborn. This matrix provides a visual breakdown of correct and incorrect classifications for each class, further illustrating the class imbalance in performance.
Model Interpretability (Grad-CAM):

Grad-CAM is implemented to visualize which parts of an image the model focused on when making a prediction.
Functions make_gradcam_heatmap and display_gradcam are created to compute the heatmap and overlay it on the original image.
This technique provides insight into the model's decision-making process and helps verify if it's focusing on medically relevant areas in the X-rays.
The result is displayed for a selected test image, showing the original X-ray and the overlaid heatmap highlighting important regions.
In essence, the project takes chest X-ray images, preprocesses them, uses a transfer learning approach with ResNet50 to build a classifier, trains and evaluates the model, and finally provides tools (classification report, confusion matrix, Grad-CAM) to understand the model's performance and how it makes predictions. The evaluation reveals overfitting, which is a key finding from the project's execution.
