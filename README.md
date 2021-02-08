# Credit_Risk_Analysis

## Project Overview: 

The goal of this analysis is to:

•	Employ different techniques to train and evaluate models with unbalanced classes.

•	Use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

The dataset from LendingClub was resampled as follow:

•	Oversampled using `RandomOverSampler` and `SMOTE` algorithms.

•	Undersampled using `ClusterCentroids` algorithm.

•	Over-and-undersampled *(a combinational approach)* using `SMOTEENN` algorithm.

Two ensemble methods were compared: `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` in order to decrease bias, variance, and to predict credit risk. 

## Results: 

### Resampling 

Resampling is the method of using samples of a dataset to improve the accuracy and quantify the uncertainty of a population parameter. 

#### Oversampling

##### Naive Random Oversampling

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/ros_results.PNG)
 
Breaking down the confusion matrix:

True Positive (TP) = 57

False Positive (FP) = 6047

False Negative (FN) = 30

True Negative (TN) = 11071

**Precision:** What proportion of positive identifications was correct?

In this model 6104 cases were predicted “high-risk” when only 57 were in fact “high-risk”. The precision is therefore 57/6104, or 0.01.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 57 were predicted correctly. The recall is 57/87, or 0.66.

**Balanced Accuracy Score:** 

Balanced Accuracy = (Sensitivity + Specificity) / 2

Balanced Accuracy = (((TP/(TP+FN)+(TN/(TN+FP))) / 2

Balanced Accuracy = (0.66 + 0.65) / 2

Balanced Accuracy = 0.65 (also shown on the screenshot above)

**F1 Score:**

F1 = 2 * (Precision * Recall) / (Precision + Recall)

F1 = 2 * (0.0066) / (0.67)

F1 = 0.02 (shown on screenshot)


##### SMOTE Oversampling

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/smote_results.PNG)

Breaking down the confusion matrix:

True Positive (TP) = 49

False Positive (FP) = 4955

False Negative (FN) = 38

True Negative (TN) = 12163

**Precision:** What proportion of positive identifications was correct?

In this model 5004 cases were predicted “high-risk” when only 49 were in fact “high-risk”. The precision is 0.01.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 49 were predicted correctly. The recall is 49/87, or 0.56.

**Balanced Accuracy Score:** 

Balanced Accuracy = 0.64

**F1 Score:**

F1 = 0.02 (shown on screenshot)


#### Under-sampling 

##### ClusterCentroids

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/undersampling_results.PNG)


Breaking down the confusion matrix:

True Positive (TP) = 50

False Positive (FP) = 9406

False Negative (FN) = 37

True Negative (TN) = 7712

**Precision:** What proportion of positive identifications was correct?

In this model 9456 cases were predicted “high-risk” when only 50 were in fact “high-risk”. The precision is therefore 50/9406, or 0.01.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 50 were predicted correctly. The recall is 50/87, or 0.57.

**Balanced Accuracy Score:** 

Balanced Accuracy = (Sensitivity + Specificity) / 2

Balanced Accuracy = (0.57 + 0.45) / 2

Balanced Accuracy = 0.51 

**F1 Score:**

F1 = 0.01

#### Combination (Over and Under) Sampling

##### SMOTEENN

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/smoteenn_results.PNG)


Breaking down the confusion matrix:

True Positive (TP) = 58

False Positive (FP) = 7432

False Negative (FN) = 29

True Negative (TN) = 9686

**Precision:** What proportion of positive identifications was correct?

In this model 7490 cases were predicted “high-risk” when only 58 were in fact “high-risk”. The precision is 0.01.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 58 were predicted correctly. The recall is 58/87, or 0.67.

**Balanced Accuracy Score:** 

Balanced Accuracy = 0.62 

**F1 Score:**

F1 = 0.02


### Ensemble Learners

#### Balanced Random Forest Classifier 

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/forest_result.PNG)

 
Breaking down the confusion matrix:

True Positive (TP) = 55

False Positive (FP) = 1659

False Negative (FN) = 32

True Negative (TN) = 15459

**Precision:** What proportion of positive identifications was correct?

In this model 1714 cases were predicted “high-risk” when only 55 were in fact “high-risk”. The precision is 55/1714, or 0.03.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 55 were predicted correctly. The recall is 0.63.

**Balanced Accuracy Score:** 

Balanced Accuracy = 0.77

**F1 Score:**

F1 = 0.06 


#### Easy Ensemble AdaBoost Classifier

![](https://github.com/KatiuscaQ/Credit_Risk_Analysis/blob/main/Resources/ada_results.PNG)

Breaking down the confusion matrix:

True Positive (TP) = 78

False Positive (FP) = 960

False Negative (FN) = 9

True Negative (TN) = 16158

**Precision:** What proportion of positive identifications was correct?

In this model 1038 cases were predicted “high-risk” when only 78 were in fact “high-risk”. The precision is therefore 78/1038, or 0.08.

**Recall (Sensitivity):** What proportion of actual positives was identified correctly?

In this study 87 cases were “high-risk” and 78 were predicted correctly. The recall is 0.90.

**Balanced Accuracy Score:** 

Balanced Accuracy = (Sensitivity + Specificity) / 2

Balanced Accuracy = (0.90 + 0.94) / 2

Balanced Accuracy = 0.92

**F1 Score:**

F1 = 0.14


## Summary:

In this study the ensemble methods both perform -as intended- better than the resampling classifiers. The results shown above indicate that the Easy Ensemble AdaBoost Classifier was the best out of the six algorithms perform for this study.

Analyzing the results of the last classifier the results are as follow:

Precision: 8% of the positive identification was correct, this means 8% were “True Positives” when 92% of the predicted “high-risk” were in fact “low-risk”. This seems not reliable.

Recall: 90% of the actual “high-risk” cases were predicted with the model. This seems reliable.

Balanced accuracy: 92% this result shows how good this binary classifier is (specially with the dataset studied which is very imbalanced).

F1: 0.14 out of all the other F1 scores this one was the higher yet is considered a low F1 score (closer to 0 than closer to 1). There is still a big number of False Positives in this model (960).

In conclusion none of the models are recommended to predict credit risks. 

It's good to remember that all models are wrong, but some are useful. 

None of the models performed in this study show reliable results to be trusted, but all the models helped in understanding the dataset in a better way.


 

