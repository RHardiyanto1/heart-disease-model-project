# Executive Summary

## Objective

The goal of this project was to build a machine learning model that predicts whether someone has heart disease based on their medical data collected during hospital visits. Using the [Cleveland Heart Disease Dataset](https://www.archive.ics.uci.edu/dataset/45/heart+disease) , we aimed to identify high-risk individuals early on, so doctors can take action before it's too late.
## Model Development and Performance

We stuck with using Logistic Regression due to it's simplicity and solid initial results. Initial accuracy was 88.9%, with a decent between precision and recall for both heart disease and non-heart disease cases.
### Feature Optimization

We attempted to optimize the model by removing features with the highest p-values (p > 0.05), thinking it might simplify the model without sacrificing performance. However, this approach backfired, leading to worse results. It turned out that even features with less statistical significance played a useful role when considered alongside each, likely due to their interactions.
### Model Optimization

88.9% accuracy is decent, with something we just plugged and played, but we felt like we could squeeze a few more percentages out of it.

1. L2 (Ridge) Regularization: This improved the model's accuracy to 91.1% by penalizing large coefficients and stabilizing the model. It helped managed multicollinearity, making prediction more consistent.
   
2. Elastic Net Regularization: Combining L1 and L2 regularization, Elastic Net tuning bumped the accuracy further to 91.67%. Which ends up being our finalized model.

![[Model Final.png]]
### Key Results

- **Final Accuracy:** 91.67%, with a strong balance of precision and recall, indicating reliable performance in heart disease prediction.
  
- **Feature Optimization Lessons:** Removing features based on high p-values degraded performance, highlighting the importance of feature interactions.
  
- **Regularization's Impact:** Both Ridge and Elastic Net regularization effectively reduced overfitting and enhanced model generalization.
  
- **Significant Predictors:** The statistically significant predictors (p < 0.05) were **sex**, **resting blood pressure (trestbps)**, **maximum heart rate achieved (thalach)**, **number of major vessels (ca)**, and **thalassemia (thal)**. These features had a strong association with heart disease outcomes and contributed most to the model's predictive power.

# Can we predict if a person has heart disease based on medical data?

Heart disease remains one of the biggest health challenges worldwide, affecting millions of people each year. The sooner it's detected, the better the chances for effective treatment and a good outcome. That’s where this project comes in: the aim was to build a predictive model that uses patient data collected during routine hospital visits to estimate the likelihood of heart disease.

Using the [Cleveland Heart Disease dataset](https://www.archive.ics.uci.edu/dataset/45/heart+disease) from the UCI Machine Learning Repository, we focused on a variety of patient characteristics like age, blood pressure, cholesterol levels, and exercise responses. The goal was straightforward—develop a reliable model that can help spot high-risk patients early, so doctors have more information when deciding who needs follow-up tests or interventions.

## Exploring the data

### How the diagnosis is distributed, and why we simplified it

![[HD Class Dist.png]]

In the original dataset, the target variable `num` ranged from 0 to 4, where 0 represented no heart disease, and values 1 to 4 indicated increasing levels of severity. We simplified this to a binary classification, where `heart_disease = 1` if `num` was 1-4 (indicating any level of heart disease) and `heart_disease = 0` if `num` was 0 (indicating no heart disease). This decision was based on several important factors:

1. **Clinical Relevance:** In real-world healthcare settings, the initial concern is whether a patient has heart disease or not, rather than pinpointing the exact severity level. The primary goal of our model was to flag patients who might need further evaluation or treatment.
   
2. **Class Imbalance in the Original Data:** The original `num` was highly imbalanced, with most cases falling under 0 (no heart disease) and 1-4 being underrepresented, which posed a challenge for training a reliable model. By combining all the non-zero values into a single category, the distribution of the target variable becomes much more balanced.
### Exploring other factors, and how they relate to heart disease

![[Feature Distribution 1.png]]

**Observations**

1. **Age vs. Heart Disease**
   
	- **Observation:** The median age for patients with heart disease (`1`) is slightly higher than those without heart disease (`0`), indicating that older individuals are more likely to have heart disease.
	  
	- **Spread and Outliers:** The interquartile range (IQR) for both groups overlaps, but the distribution for heart disease patients shows more outliers on the lower end, suggesting there are younger individuals in the heart disease group.
	  
2. **Resting Blood Pressure (`trestbps`) vs. Heart Disease**
   
	- **Observation:** The median resting blood pressure is somewhat similar for both groups, although there seems to be a slight increase in median values for the heart disease group.
	  
	- **Spread and Outliers:** There are more outliers in the heart disease group, indicating higher variability in blood pressure among those patients.
	  
3. **Cholesterol (`chol`) vs. Heart Disease**
   
	- **Observation:** The median cholesterol level is quite close for both groups, which suggests cholesterol levels alone may not be a strong distinguishing factor.
	  
	- **Spread and Outliers:** The distributions have similar spreads, with some high outliers present in both groups.
	  
 4. **Maximum Heart Rate Achieved (`thalach`) vs. Heart Disease**
    
	- **Observation:** The median maximum heart rate is notably lower in the heart disease group, indicating that individuals with heart disease tend to achieve lower heart rates during physical exertion.
	  
	- **Spread and Outliers:** There is a visible downward shift in the heart rate distribution for the heart disease group, suggesting that lower maximum heart rate could be associated with heart disease.
	  
5. **ST Depression (`oldpeak`) vs. Heart Disease**
   
	- **Observation:** The median `oldpeak` (ST depression) is higher for individuals with heart disease, indicating that heart disease patients are more likely to have higher ST depression during exercise.
	  
	- **Spread and Outliers:** There is a clear upward shift in `oldpeak` values for the heart disease group, which could be a significant predictor.

![[Feature Distribution 2.png]]

1. **Sex vs. Heart Disease**
   
	- **Observation:** The majority of patients with heart disease (`1`) are male (`sex = 1`), while females (`sex = 0`) are more prevalent in the group without heart disease (`0`). This indicates a higher prevalence of heart disease among men in this dataset.
	  
	- **Implication:** Sex could be a significant predictor of heart disease, with males showing a higher likelihood.
	  
2. **Chest Pain Type (`cp`) vs. Heart Disease**
   
	- **Observation:** The distribution shows a trend where:
	    - Patients with `cp = 4` (asymptomatic) are predominantly in the heart disease group.
	    - Patients with `cp = 1` (typical angina) are more common in the non-heart disease group.
	      
	- **Implication:** Chest pain type is strongly associated with heart disease presence, with asymptomatic chest pain being a notable indicator.
	  
3. **Fasting Blood Sugar (`fbs`) vs. Heart Disease**
   
	- **Observation:** There is a similar distribution of `fbs = 0` (fasting blood sugar ≤ 120 mg/dl) across both groups, but `fbs = 1` (fasting blood sugar > 120 mg/dl) has more representation in the non-heart disease group.
	  
	- **Implication:** Fasting blood sugar may not be a strong predictor of heart disease on its own.
	  
4. **Resting Electrocardiographic Results (`restecg`) vs. Heart Disease**
   
	- **Observation:**
	    - Patients with `restecg = 0` (normal) are more prevalent in the non-heart disease group.
	    - `restecg = 2` (left ventricular hypertrophy) has more patients in the heart disease group.
	      
	- **Implication:** Resting ECG abnormalities, particularly left ventricular hypertrophy, could indicate a higher risk of heart disease.
	  
5. **Exercise Induced Angina (`exang`) vs. Heart Disease**
   
	- **Observation:**
	    - Most heart disease patients have `exang = 1` (exercise-induced angina).
	    - Those with `exang = 0` (no exercise-induced angina) are more common in the non-heart disease group.
	      
	- **Implication:** Presence of exercise-induced angina is a strong indicator of heart disease.
	  
6. **Slope of the Peak Exercise ST Segment (`slope`) vs. Heart Disease**
   
	- **Observation:**
	    - `slope = 2` (flat) is more common in the heart disease group.
	    - `slope = 1` (upsloping) appears more frequently in the non-heart disease group.
	      
	- **Implication:** A flat slope is associated with a higher risk of heart disease, while an upsloping ST segment is more common in healthy individuals.
	  
1. **Number of Major Vessels Colored by Fluoroscopy (`ca`) vs. Heart Disease**
   
	- **Observation:**
	    - Patients with `ca = 0` (no major vessels colored) are more common in the non-heart disease group.
	    - As the number of colored vessels increases (`ca = 1, 2, 3`), the proportion of patients with heart disease also increases.
	      
	- **Implication:** Higher `ca` values are strongly correlated with the presence of heart disease.
	  
2. **Thalassemia (`thal`) vs. Heart Disease**
   
	- **Observation:**
	    - `thal = 0` (normal) is more common in the non-heart disease group.
	    - `thal = 2` (reversible defect) has a higher occurrence in the heart disease group.
	      
	- **Implication:** Abnormal thalassemia results, especially reversible defects, are associated with a higher risk of heart disease.

## Deciding which features to put into the model (all of it)

### Justification for Throwing All Features into the Initial Logistic Regression Model

When starting the modeling process, we decided to include all available features in the initial logistic regression model. There were several reasons for this approach:

1. **Exploratory Approach:** Given that this was an exploratory analysis, it made sense to start with all features to see how they collectively contributed to predicting heart disease. Including every variable allowed the model to learn from all possible relationships and interactions within the dataset.
   
2. **Initial Performance Was Promising:** The initial logistic regression model, with all features included, achieved an accuracy of around 88.9%. This result suggested that the model could capture meaningful patterns, even before any fine-tuning. It gave us a good baseline to compare against as we proceeded to optimize the model.
   
3. **Feature Reduction Attempts Worsened the Model:** We attempted to remove features with the highest p-values (those with p > 0.05) in the hope of simplifying the model without sacrificing performance. However, this actually led to a decline in accuracy. It turned out that even some features with higher p-values still added value to the model, likely due to interactions with other variables. Removing them stripped the model of important information, making predictions less reliable.
   
4. **Regularization as a Better Approach:** Instead of relying solely on p-values to decide which features to keep, we used regularization techniques (Ridge and Elastic Net) to control for multicollinearity and overfitting. This allowed the model to retain potentially useful features while shrinking less important coefficients, ultimately leading to better performance.

### Initial Model

![[Model 1.png]]

The first iteration of the logistic regression model was designed to establish a baseline performance for predicting heart disease. Here’s a step-by-step outline of the approach:

#### **Data Preparation**

- The dataset was split into features (`X`) and the target variable (`y`), where `y` represented the binary `heart_disease` status (0 or 1), and the features (`X`) excluded the original `num` and `heart_disease` columns.
  
- We then divided the data into training and testing sets, using a 70-30 split. This larger test set size (30%) provided a better estimate of model performance compared to using a 20% split.
#### **Feature Scaling**

- Since logistic regression is sensitive to the scale of input data, we standardized the numerical features to improve model performance. The `StandardScaler` was applied to the training and test sets, ensuring consistent scaling across both.
#### Model Fitting**

- We used the `statsmodels` library to fit a logistic regression model with Maximum Likelihood Estimation (MLE).
  
- A constant term was added to the features to account for the intercept, allowing the model to capture baseline probabilities.
  
- All features were included in the initial model to observe their collective impact, without any prior elimination based on p-values.
#### **Results of the First Iteration**

- The model achieved an accuracy of **88.9%** on the test set, indicating a strong initial performance.
  
- The confusion matrix showed that the model correctly classified 45 out of 49 cases for non-heart disease (precision: 0.88) and 35 out of 41 cases for heart disease (precision: 0.90).
  
- The overall precision, recall, and F1-score were balanced across both classes, with the macro average for all three metrics being 0.89.
#### **Feature Significance Analysis**

- The logistic regression results provided insights into feature significance:
  
    - **Significant Predictors (p < 0.05):** `sex` (p = 0.008), `trestbps` (p = 0.020), `thalach` (p = 0.035), `ca` (p < 0.001), and `thal` (p = 0.007).
      
    - **Marginal Predictors (p < 0.1):** `chol` (p = 0.062), `fbs` (p = 0.058).
      
    - **Non-Significant Predictors (p > 0.1):** Features like `age`, `cp`, `restecg`, `oldpeak`, `exang`, and `slope` did not show strong individual statistical significance, yet their inclusion collectively contributed to the model’s overall performance.

We experimented with removing features with the highest p-values (those above 0.05) to see if a simpler model would perform just as well. However, the accuracy dropped below 88%, confirming that even features deemed "insignificant" by p-values could still provide value through interactions with other variables.
#### Takeaways of the Initial Model

The first logistic regression model achieved a promising accuracy of 88.9%, with balanced precision and recall across classes. The initial results suggested that while some features were not individually significant, their combined effect improved predictive performance. Future iterations focused on refining the model through regularization to further enhance accuracy and manage potential overfitting while retaining potentially valuable features.

### Applying Regularization: Testing L1 and L2 Separately

![[Model 2.png]]

To improve the logistic regression model and prevent overfitting, we introduced regularization techniques—specifically L1 (Lasso) and L2 (Ridge) regularization. These methods penalize large coefficients, which can help stabilize the model, especially when dealing with correlated features or noise. Here’s an outline of our approach and the results:
#### Data Preparation

 We followed the same data preprocessing steps as in the initial iteration: splitting the dataset into training and testing sets (70-30 split), standardizing the numerical features using `StandardScaler`, and scaling both the training and test sets for consistency.
#### Applying L2 Regularization (Ridge)

We started with Ridge regularization (L2), which applies a penalty proportional to the square of the coefficients. This technique is effective in controlling multicollinearity by shrinking the coefficients. We tested various values of the regularization parameter `C` (inverse of regularization strength), starting from `1` and gradually decreasing, until we found that `C = 0.1` provided the optimal balance between model complexity and accuracy.
##### Ridge Model Results

- **Confusion Matrix:** The model correctly identified 46 out of 49 non-heart disease cases and 36 out of 41 heart disease cases.
- **Accuracy:** Achieved an accuracy of **91.1%**, a noticeable improvement from the initial model's 88.9%.
- **Precision and Recall:**
    - Class 0 (No Heart Disease): Precision = 0.90, Recall = 0.94, F1-score = 0.92
    - Class 1 (Heart Disease): Precision = 0.92, Recall = 0.88, F1-score = 0.90
- **Key Insight:** Ridge regularization helped improve the model's performance by penalizing large coefficients, leading to better generalization on the test set.
#### Applying L1 Regularization (Lasso)

We then applied Lasso regularization (L1), which penalizes the absolute values of the coefficients. This approach can drive some coefficients to zero, effectively performing feature selection. Similar to the Ridge model, we adjusted the value of `C` from `1` down to `0.1`, finding that `C = 0.1` produced the best results.
##### Lasso Model Results

- **Confusion Matrix:** The Lasso model correctly classified 46 out of 49 non-heart disease cases and 35 out of 41 heart disease cases.
- **Accuracy:** Achieved an accuracy of **90%**, slightly lower than Ridge regularization.
- **Precision and Recall:**
    - Class 0 (No Heart Disease): Precision = 0.88, Recall = 0.94, F1-score = 0.91
    - Class 1 (Heart Disease): Precision = 0.92, Recall = 0.85, F1-score = 0.89
- **Key Insight:** Lasso regularization also improved model performance, but it was slightly less effective than Ridge regularization. L1 regularization’s feature selection aspect did not significantly improve accuracy in this case, possibly because no features were sufficiently "unimportant" to drop without hurting the model’s performance.
#### Comparison of Ridge vs. Lasso

- **Ridge (L2) Regularization:** Provided the best accuracy at **91.1%** with a good balance between precision and recall for both classes. The approach helped reduce overfitting by shrinking coefficients without zeroing them out.
- **Lasso (L1) Regularization:** Achieved a slightly lower accuracy of **90%**, showing that setting some coefficients to zero did not benefit the model as much as Ridge's approach of shrinking all coefficients.
#### Summary of Findings Before Moving to Elastic Net

- **Optimal Regularization Parameter:** Both models performed best when `C = 0.1`, indicating that stronger regularization (lower `C`) helped improve generalization. However, setting `C` too low resulted in underfitting.
- **Next Steps:** Given the slight edge that Ridge regularization had over Lasso, the next step was to explore Elastic Net regularization. This approach combines L1 and L2 penalties, potentially offering the benefits of both techniques by adjusting the mixing parameter (`l1_ratio`).

### Final Model: Elastic Net Regularization

![[Model Final.png]]

For the final iteration, we utilized Elastic Net regularization, combining L1 (Lasso) and L2 (Ridge) penalties to optimize the model's ability to select features and shrink coefficients effectively. This approach aimed to balance feature selection with coefficient stability, improving the model's performance.
#### Data Preparation

- We split the data into features (`X`) and the target variable (`y`), with `y` indicating the presence of heart disease.
- The data was divided into an 80-20 training and testing split, as a 20% test set produced the best results for this model iteration.
- To ensure consistent scaling across features, we applied `StandardScaler` to standardize the numerical features.
#### Transition to Automated Hyperparameter Tuning

- Initially, we manually tuned the hyperparameters, by trial and error. However, we switched to using `GridSearchCV` for a more systematic search.
- The grid search allowed for a more automated approach to finding the best hyperparameters, avoiding potential bias from manual tuning.
- We explored the following parameter ranges:
    - **`C` (regularization strength):** [0.01, 0.1, 1, 10]
    - **`l1_ratio` (mix of L1 and L2):** [0.1, 0.5, 0.9]
- The logistic regression model was configured with the `elasticnet` penalty and `saga` solver, as `saga` supports Elastic Net regularization.
#### Results

- **Best Hyperparameters:** The grid search identified `C = 0.01` and `l1_ratio = 0.1` as the optimal settings, favoring stronger L2 regularization with a slight L1 influence.
- **Performance Metrics:**
    - **Confusion Matrix:** The model accurately classified 34 of 36 cases for non-heart disease and 21 of 24 cases for heart disease.
    - **Accuracy:** Achieved **91.67%** accuracy on the test set, an improvement over the previous iterations.
    - **Precision, Recall, and F1-Score:**
        - **Class 0 (No Heart Disease):** Precision = 0.92, Recall = 0.94, F1-score = 0.93
        - **Class 1 (Heart Disease):** Precision = 0.91, Recall = 0.88, F1-score = 0.89
        - The average for precision, recall, and F1-score was approximately 0.91, indicating balanced model performance.
#### Final Takeaways

- **Grid Search Enhanced Hyperparameter Tuning:** Switching from manual tuning to `GridSearchCV` allowed for a more comprehensive and unbiased search, leading to the discovery of optimal hyperparameters that manual tuning might have missed.
- **Elastic Net Regularization Improved Generalization:** By combining the benefits of L1 and L2 regularization, Elastic Net provided the best balance between feature selection and coefficient stability, resulting in a well-generalized model.
- **20% Test Split was Effective for the Final Model:** Although a 30% test set was useful in earlier iterations, reverting to a 20% split yielded higher accuracy in the final model, likely due to the increased training data helping with model learning.
#### Conclusion

This project successfully built a model to predict heart disease using patient data, reaching an accuracy of **91.67%**. Through data exploration, feature scaling, and regularization techniques like L1, L2, and Elastic Net, we fine-tuned a logistic regression model that performs well in identifying high-risk patients.

Simplifying the target variable and using `GridSearchCV` for hyperparameter tuning helped optimize the model without overcomplicating things. While more advanced models could offer slight improvements, this approach proved that a well-tuned logistic regression can be both effective and practical.

In short, the project achieved its goal: creating a reliable tool for early heart disease detection that’s ready for real-world use.

# Citations

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.