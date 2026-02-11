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
- Highlighted the best score in each column

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost | 0 | 0 | 0 | 0 | 0 | 0 |
| Logistic Regression | 0 | 0 | 0 | 0 | 0 | 0 |
| Random Forest | 0 | 0 | 0 | 0 | 0 | 0 |
| KNN | 0 | 0 | 0 | 0 | 0 | 0 |
| Naive Bayes | 0 | 0 | 0 | 0 | 0 | 0 |
| Decision Tree | 0 | 0 | 0 | 0 | 0 | 0 |

**Observation of each model on the Dry bean dataset:**

*TODO: add the observations once the implementation is completed*