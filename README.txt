How the Code Works

1.Images are loaded from folders, each representing a disease class. The images are passed through a
feature extraction step using ResNet50, producing compact feature vectors instead of using raw pixels.
The dataset is split into training and testing sets. A KNN classifier is trained on the features, with
hyperparameters (number of neighbors, distance metric, weighting) optimized via GridSearchCV and
cross-validation. The final model is then evaluated using classification metrics


2.Feature.Extraction
ResNet50, pre-trained on ImageNet and excluding its final classification layer, extracts meaningful visual
features from resized images. These features capture shapes and textures critical for distinguishing
diseases.
.knn


KNN uses these extracted features to classify new images by finding the majority class among the
nearest neighbors. The model’s parameters are fine-tuned to improve accuracy.
3. Model Training and Evaluation
•Training the Model:
KNN is trained on the feature vectors extracted from training images. The grid search tests various
hyperparameters (neighbors, distance, weighting) to find the best combination.
•Best Parameters Output:
After training, the best hyperparameters identified by grid search are used for the final model, ensuring
optimal classification performance.
•Evaluating the Model:
The model’s performance is assessed on the test set, measuring how well it generalizes to unseen data.
•Classification Report:
Using classification_report(), precision, recall, F1-score, and support for each disease class are
generated. Precision measures the correctness of positive predictions, recall measures how many actual
positives were found, and support indicates the number of true instances per class.
•Confusion Matrix:
The confusion matrix visually displays correct and incorrect predictions per class, with rows representing
actual classes and columns representing predicted classes. This helps identify which classes
the model confuses


Comparison Between ResNet50 and KNN:-

.ResNet50 is a deep learning model used here solely to extract important features
from images. It converts images into compact numerical vectors that represent
visual patterns. It is pre-trained and not updated during this project.
.KNN is a traditional machine learning algorithm that uses these feature vectors
to classify images by comparing them to the closest training examples. KNN learns
from the extracted features and predicts the class based on similarity.
.In short, ResNet50 handles feature extraction while KNN performs classification.
This separation simplifies the process and combines the power of deep learning
with the simplicity of KNN.


Conclusion:-
Integrating ResNet50 for feature extraction with KNN classification
yields a robust system for plant disease detection. The method
efficiently captures complex visual patterns and applies a clear, tunable
classification technique, resulting in accurate disease diagnosis
from leaf images.


## Dataset

Please download the dataset from Kaggle:  
[Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Note:**  
Feature extraction using ResNet50 and VGG16 takes considerable time and computational resources, so pre-extracted features are not included.

---

## Usage

1. Download and prepare the dataset.  
2. Run feature extraction scripts (ResNet50 and VGG16) — this may take some time.  
3. Train the KNN model on the extracted features.  
4. Predict plant disease categories using the trained model.

---

If you have any questions or need help, feel free to open an issue or contact me.