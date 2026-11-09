# Heart Disease Classification – ML Assignment 2

## a. Problem Statement
Predict the presence of heart disease (binary: 0 = no, 1 = yes) from 13 clinical attributes.

## b. Dataset Description
Source: UCI / Kaggle Heart Disease dataset (heart.csv).
Rows: ___ (after removing duplicates). Features: 13 (age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal). Target: `target` (0/1).

## c. GitHub Repository Link
https://github.com/2026aj05657-cmd/ml-assignment-2/
https://ml-assignment-2-2ccwrmpl6lpjvqfblouccv.streamlit.app/

## d. Models Used
| ML Model Name      | Accuracy | AUC | Precision | Recall | F1 | MCC | <br>
Logistic Regression	     0.8026	0.8927	0.8250	0.8049	0.8148	0.6038 <br>
Decision Tree	           0.7368	0.7958	0.7692	0.7317	0.7500	0.4732 <br>
KNN	                     0.8289	0.8847	0.8182	0.8780	0.8471	0.6557 <br>
Naive Bayes	             0.8026	0.8794	0.8095	0.8293	0.8193	0.6022 <br>
Support Vector Machine	 0.7895	0.8760	0.7778	0.8537	0.8140	0.5760 <br>


## Observations


ML Model Name and	Observation about model performance
1. Logistic Regression	
Highest AUC (0.893) and highest precision (0.825), so it ranks patients by risk better than any other model. Its recall (0.805) is lower than KNN's, so it misses a few more patients who have the disease. Its MCC of 0.604 is solid. It's a strong, interpretable baseline, which suggests the features are largely linearly separable.
2. Decision Tree	
Weakest model on every metric (accuracy 0.737, AUC 0.796, MCC 0.473). Even with depth limited to 5, a single tree overfits the small de-duplicated training set and gives coarse probability estimates, which explains the low AUC.
3. KNN	
Best accuracy (0.829), recall (0.878), F1 (0.847) and MCC (0.656), with AUC (0.885) close behind Logistic Regression. With scaled features and k=7, nearby patients carry useful information. Its high recall matters most in a medical setting, where missing a patient with heart disease is the costliest error.
4. Naive Bayes	
Performs almost the same as Logistic Regression (accuracy 0.803, MCC 0.602) and has better recall (0.829), even though it assumes the features are independent, which they aren't. It's the simplest and fastest model, and it stays competitive.
5. Support Vector Machine	
Good recall (0.854), but the lowest precision (0.778) apart from the Decision Tree, which pulls its MCC down to 0.576. The RBF kernel didn't add anything over the linear model on this small dataset. It might improve with tuning of C and gamma.

Overall winner	KNN. It has the highest MCC (0.656), F1, accuracy and recall. MCC is the main criterion because it balances all four cells of the confusion matrix. Logistic Regression comes a close second, with the best AUC and better interpretability.

## How to run
```
pip install -r requirements.txt
python train.py        # needs heart.csv in this folder
streamlit run app.py
```
