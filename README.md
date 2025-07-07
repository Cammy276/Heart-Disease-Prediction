# Heart Disease Prediction Models Comparison Using Different Algorithms
Assignment for UECS3483 Data Mining

This repository presents a comparative machine learning study using different algorithms for heart disease prediction based on patient medical records. It includes code for data preprocessing, model training, overfitting evaluation, performance evaluation, and final model saving using joblib.

## Objective
The objective of this project is to compare multiple machine learning algorithms to identify the best perfoming model for predicting heart disease. At the data preprocessing, two version of dataset are kept, which are dataset with outliers and dataset without outliers, to assess the impact of outliers on model performance. To evaluate and reduce the risk of overfitting, cross-validation and accuracy scores are applied. The models are evaluated using using performing matrics including precision, recall, F1-score, accuracy, AUC score, and confusion matrics. The model with highest AUC score is selected as final model and saved for future application.

## Technologies Used
|Tool/Library|Description|
|---|---|
|Google Colab| Cloud-based platform for running Python code.|
|Python|Programming language used for data preprocessing, model building, and saving.|
|pandas|For handling structured data and dataframes.|
|numpy|For numerical operations.|
|matplotlib|For plotting data and creating visualizations.|
|seaborn|For advanced statistical data visualization.|
|scikit-learn|For encoding, scaling, splitting data, and preprocessing pipelines.|
|scipy.stats|For statistical analysis.|
|statsmodels|Used to compute Variance Inflation Factor (VIF) to detect multicollinearity.|
|scikit-learn|Provides tools for preprocessing, traditional machine learning models, cross-validation, evaluation metrics, pipelines, and hyperparameter tuning.|
|scikit-learn.metrics|For performance evaluation metrics like accuracy, precision, recall, F1-score, ROC curve, AUC, and confusion matrix.|
|scikit-learn.model_selection|For train-test splitting, cross-validation, and hyperparameter tuning.|
|scikit-learn.ensemble|Includes ensemble models like Random Forest, AdaBoost, Gradient Boosting, and Voting Classifier.|
|scikit-learn.linear_model|For logistic regression.|
|scikit-learn.tree|For decision tree models.|
|scikit-learn.svm|For support vector machines (SVC).|
|scikit-learn.naive_bayes|For Naive Bayes classifier.|
|scikit-learn.neural_network|For multilayer perceptron (MLP) classifier.|
|scikit-learn.neighbors|For k-nearest neighbors (KNN) classifier.|
|xgboost|For gradient boosting models.|
|lightgbm|For efficient gradient boosting on decision trees.|
|catboost|For categorical boosting.|
|joblib|For saving the final trained model.|
|warnings|Used to suppress warnings for cleaner output.|


## Algorithms Used
1. AdaBoost
2. CatBoost
3. Decision Tree
4. GradientBoosting
5. k--Nearest Neighbors (KNN)
6. LightGBM
7. Logistic Regression 
8. Multilayer Perceptron (MLP)
9. Naïve Bayes
10. Random Forest 
11. Support Vector Machine (SVM)
12. VotingClassifier
13. XGBoost
14. Voting (Random Forest, LightGBM, XGBoost)

## Methodology
1. Data collection
    - Total of 1027 records  
    - Total of 14 features
        - 13 independent features and 1 dependent feature
        - 8 categorical features (7 nominal features and 2 ordinal features) and 5 numerical features (4 ratio features and 1 interval features) 
3. Exploratory Data Analysis (EDA) on Original Dataset
4. Data splitting
    - 70% Training data
    - 15% Validation data
    - 15% Testing data
5. Data preprocessing
    - Handling duplicates
        - Removing duplicated records
    - Handling Missing values
        - Removing null values
        - Finding best parameters by RandomizedSearchCV
        - Imputation missing values using Random Forest Algorithm
     - Handling Outliers
         - Analyzing skewness in numerical features
         - Applying Yeo-Johnson transformation (for absolute skewness > 1)
         - Identifying outliers using Z-score (±3 standard deviations)
         - Capping outliers using 5th and 95th percentiles
         - Saving both original (with outliers) and capped (without outliers) datasets
6. Feature scaling
    - Applying standardization (z-score scaling) to numerical features
7. Feature Importance Visualization
    - Using Random Forest to visualize feature importance on both datasets (with and without outliers)
8. Categorical features encoding
    - One-hot encoding for nominal categorical features
    - Ordinal encoding for ordinal categorical features
9. Dataset Selection (With vs. Without Outliers)
    - Selecting 3 base algorithms
    - Performing hyperparameter tuning using HalvingRandomSearchCV
    - Evaluating risk of overfitting by 3-fold cross-validation
    - Comparing performance metrics (precision, recall, F1, accuracy, AUC) and confusion matrics
    - Selecting the better-performing dataset for final modeling
10. Final Model Training
    - Training the remaining algorithms using the selected dataset
    - Performing hyperparameter tuning using HalvingRandomSearchCV and applying the best parameters on these models
    - Evaluating risk of overfitting by 3-fold cross-validation
    - Evaluating performance metrics (precision, recall, F1, accuracy, AUC) and confusion matrics
11. Building voting model by top 3 models based on validation performance
    - Evaluating risk of overfitting by 3-fold cross-validation
    - Evaluating performance metrics (precision, recall, F1, accuracy, AUC) and confusion matrics
12. Final Evaluation on Testing Set
    - Comparing all models using ROC curves and AUC scores
    - Selecting the model with the highest AUC as the final model
13. Model Saving
    - Saving the best model using joblib for future application


## Google Colab Link
1) Part 1: https://colab.research.google.com/drive/1NDyaXNd7FHiQRihu1Ll3bN3VJFJu_IzL?usp=sharing
2) Part 2 & 3: https://colab.research.google.com/drive/1gHpx92KScLWY7mMaqZQ8aqTz0gIWCRwP?usp=sharing

## Authors
- [@Yu-2008] (https://github.com/Yu-2008)
- [@Cammy276] (https://github.com/Cammy276)
- [@LIOWKEHAN] (https://github.com/LIOWKEHAN)
  

## Contributing

Contributions are always welcome!

To get started:

1. **Fork** the repository to your GitHub account.
2. **Create a new branch** for your feature or fix:  
   `git checkout -b your-feature-name`
3. **Make your changes** and commit them with a clear message:  
   `git commit -m "Add: Description of your change"`
4. **Push** your branch to your forked repository:  
   `git push origin your-feature-name`
5. **Open a Pull Request** from your branch to the main project.

Feel free to open an issue first if you'd like to discuss your idea before implementing it.
