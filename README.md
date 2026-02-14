# Dry Bean Classification using Machine Learning models
Implementation of multiple classification models to classify 'Dry Bean' and comparing them based on evalution metrics using Machine Learning

# Problem Statement
The objective of this project is to develop an automated machine learning system capable of classifying seven different registered varieties of dry beans based on their physical characteristics. 

Many bean varieties are visually similar, making manual classification difficult, time-consuming, and prone to human error. This project leverages a dataset of 13,611 grains processed through computer vision to identify the most effective classification algorithm—ranging from linear models like Logistic Regression to complex ensembles like XGBoost—to ensure a reliable, automated seed classification process.

# Dataset description
The dataset is composed of features extracted from high-resolution images of dry beans. A computer vision system was used for segmentation and feature extraction, resulting in 16 morphological features for each grain.

General Information:
1. Total Instances: 13,611
2. Total Attributes: 17 (16 numeric features + 1 categorical target)
3. Classes: 7 (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, and Sira)

Attribute Breakdown:
The 17 features consists of 12 dimensions, 4 shape forms and 1 target class, whcih were obtained from the grains:
1. Area (A): The area of a bean zone and the number of pixels within its boundaries.
2. Perimeter (P): Bean circumference is defined as the length of its border.
3. Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
4. Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
5. Aspect ratio (K): Defines the relationship between L and l.
6. Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
7. Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
8. Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
9. Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10. Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11. Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12. Compactness (CO): Measures the roundness of an object: Ed/L
13. ShapeFactor1 (SF1): Specific mathematical descriptors for geometric nuances.
14. ShapeFactor2 (SF2): Specific mathematical descriptors for geometric nuances.
15. ShapeFactor3 (SF3): Specific mathematical descriptors for geometric nuances.
16. ShapeFactor4 (SF4): Specific mathematical descriptors for geometric nuances.
17. Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)

# Models Used:
This project benchmarks six distinct machine learning algorithms:
1. Logistic Regression (Linear Model):
    - Predicts the probability of a class by fitting data to a logistic (sigmoid) function.
2. XGBoost (Gradient Boosting):
    - An advanced ensemble method that builds decision trees sequentially. Each new tree attempts to correct the errors (residuals) made by the previous ones.
3. Random Forest (Bagging Ensemble):
    - Constructs a "forest" of multiple decision trees, each trained on a random subset of data. The final prediction is based on the majority vote of all trees.
4. K-Nearest Neighbors (KNN):
    - A distance-based algorithm that classifies a bean based on the most common variety among its $k$ closest neighbors in the feature space.
5. Naive Bayes (Probabilistic):
    - Based on Bayes' Theorem, it calculates the probability of each class assuming that all input features are independent of each other.
6. Decision Tree (Non-Linear):
    - Uses a flowchart-like structure to split the data into branches based on feature values (e.g., Is Area > 50,000?).

**Comparison Table with the evaluation metrics for all the 6 models:**
| Model               |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|---------------------|------------|--------|-------------|----------|------------|--------|
| Logistic Regression |     0.9207 | 0.9934 |      0.9214 |   0.9207 |     0.9208 | 0.9041 |
| Decision Tree       |     0.8898 | 0.9320 |      0.8896 |   0.8898 |     0.8896 | 0.8669 |
| kNN                 |     0.9152 | 0.9811 |      0.9158 |   0.9152 |     0.9153 | 0.8974 |
| Naive Bayes         |     0.8979 | 0.9902 |      0.9005 |   0.8979 |     0.8980 | 0.8772 |
| Random Forest       |     0.9192 | 0.9910 |      0.9194 |   0.9192 |     0.9191 | 0.9023 |
| XGBoost             |     0.9280 | 0.9939 |      0.9282 |   0.9280 |     0.9280 | 0.9129 |

**Observation of each model on the Dry bean dataset:**
| Model Name          |   Observation about model performance     |
|---------------------|-------------------------------------------|
| Logistic Regression |  High performance, indicating many features have strong linear correlations to the target classes.  |
| Decision Tree       |  Stable: Uses clear physical "if-then" rules. Consistent performance across all data sizes.  |
| kNN                 |  Performed well on the large training set where data density is high, though sensitive to local noise.  |
| Naive Bayes         |  Assumption Error: Incorrectly assumes bean features (like Area/Perimeter) are independent.  |
| Random Forest       |  Excellent stability and class separation (high AUC); effectively reduces variance through bagging. |
| XGBoost             |  Best model: Highest accuracy and MCC. Effectively captures non-linear relationships using gradient boosting.  |
