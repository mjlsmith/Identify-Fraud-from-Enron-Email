# Enron Submission Free-Response Questions






### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The main goal of this project was to use Machine learning algorithms to investigate persons of interest within the Enron Dataset. The dataset consisted of emails from employees of Enron as well as financial data. The dataset was also broken up to distinguish "POI" (Persons of interest) who, per the project [details](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923), were "individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity."

The most noticable outlier was the TOTAL row. This had to be removed otherwise it would sku all of the data. K. Lay and J. Skilling were finincial outliers but they were left in because they were needed in the investigation.

### What features did you end up using in your POI identifier
`exercised_stock_options`  `shared_receipt_with_poi` `fraction_from_poi` `expenses other salary`

I also created the following features:

`fraction_from_poi` (The fraction of messages to that person from a person of interest)
`fraction_to_poi` (The fraction of messages from that person to a person of interest)
`from_specific_email` 
### These were made under the assumption that a POI would email another POI more often

I went through several rounds of feature selection. . On the first step I created set of features based on data visualization and intuition. Then I examine three classifications on these features. Decision Trees was determined to be the strongest algorithm so that was the one I chose. Since I choose Decision Trees as a classifier, I used the feature importance method to optimize features. My results were as follows:

| rank | feature|
|-------|------|
| 0.031483 | salary |
| 0.145820 | fraction_from_poi|
| 0.185831 | expenses |
| 0.195282 | shared_receipt_with_poi |
| 0.217197 | exercised_stock_options|
| 0.224388 | other|






### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
The algorithms I tied were Naive Bayes (Accuracy 0.82100, Precision 0.29026, Recall 0.23700) SVM (ERROR) and Decision Tree (Accuracy 0.82620, Precision 0.34617, Recall 0.34150)

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  

Basically, tuning the parameters means adjusting them to get the best performance from the algorithm. If this isn't done properly, your results may be flawed or you may not use the best set of parameters for your dataset. 

### How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

I used [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to tune my algorithm.

The results were as follows:

| Parameter | Settings for Investigation |
|----------|-------------|
|min_samples_split |	[2,6,8,10] |
|Splitter	| (random,best) |
| max_depth	| [None,2,4,6,8,10,15,20] |


### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

To validate my analysis I used Stratified Shuffle Split. This can be seen in the sklearn [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

For evaluation metric, I used percision an recall. The final results can be seen below. 

| Metric | Value |
|--------|-------|
|percision|0.52166|
|Recall | 0.41550|
|Accuracy | 086207|
| True Positive| 831|
|False Positive| 762|
| True Negitive | 11238|

### Conclusion 
Percision and Recall have very similar values and both are higher than 0.3 so the goal was reached. A percision of 0.52166 means that when the model detects a person as a POI, in 52% of the cases, it is correct. Also with Recall POI's were detected 41%. While these may not seem like very good results, because the dataset is far from perfect, and horribly incomplete, the results, I feel, are "good enough". 

# Average Performance 
## Algorithm outputs log 
* STEP 1. Init stage. Select classificator 
  * features_list: features_list = ['poi', 'fraction_to_poi', 'fraction_from_poi',    'from_specific_email', 'from_messages', 'exercised_stock_options', 'shared_receipt_with_poi', 'expenses', 'other', 'bonus', 'salary', 'total_stock_value']
  * Classificator: clf = tree.DecisionTreeClassifier() 
  * Metrics: Accuracy: 0.82620	
  * Precision: 0.34617	
  * Recall: 0.34150	
  * F1: 0.34382	
  * F2: 0.34242 
  * Total predictions: 15000 
  * True positives: 683	
  * False positives: 1290 
  * False negatives: 1317	
  * True negatives: 11710 
  * Classificator: clf = GaussianNB() 
  * Metrics: Accuracy: 0.82100	
  * Precision: 0.29026	
  * Recall: 0.23700	
  * F1: 0.26094
  * F2: 0.24603 
  * Total predictions: 15000 
  * True positives: 474	
  * False positives: 1159 
  * False negatives: 1526	
  * True negatives: 11841
* STEP2. Select features by Decision Trees feature_importances_ 
  * feature_importances_ Rank of features 0.260361 : other 0.231356 : exercised_stock_options 0.225797 : expenses 0.134677 : fraction_from_poi 0.118998 : shared_receipt_with_poi 0.028810 : salary 0.000000 : fraction_to_poi 0.000000 : from_specific_email 0.000000 : from_messages 0.000000 : bonus 0.000000 : total_stock_value
 * Metrics after optimizing 
   * Accuracy: 0.84029	
   * Precision: 0.43873	
   * Recall: 0.42250	
   * F1: 0.43046	F2: 0.42565 
   * Total predictions: 14000 
   * True positives: 845	
   * False positives: 1081 
   * False negatives: 1155	
   * True negatives: 10919

* STEP3. Tune the algorithm 
  * features_list features_list = ['poi', 'salary', 'fraction_from_poi', 'exercised_stock_options', 'shared_receipt_with_poi', 'expenses', 'other'] 
  * best estimator: DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=6, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=12, random_state=None, splitter='best') 
 * Metrics after tuning 
   * Accuracy: 0.86207 
   * Precision: 0.52166 
   * Recall: 0.41550 
   * F1: 0.46257	
   * F2: 0.43313 
   * Total predictions: 14000 
   * True positives: 831	
   * False positives: 762 
   * False negatives: 1169	
   * True negatives: 11238

* STEP4. Change features by hand (examine only email features) 
  * features_list = ['poi', 'fraction_from_poi', 'fraction_to_poi',
'exercised_stock_options', 'shared_receipt_with_poi']
  * Metrics Accuracy: 0.83108	
  * Precision: 0.43510	
  * Recall: 0.32850	
  * F1: 0.37436	
  * F2: 0.34543 
  * Total predictions: 13000 
  * True positives: 657	
  * False positives: 853 
  * False negatives: 1343	
  * True negatives: 10147

* STEP5. Tune parameters by hand 
  * parameters clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=2,max_depth=2, splitter='best') 
   * Metrics Accuracy: 0.83550	
    * Precision: 0.37947	
    * Recall: 0.23850	
    * F1: 0.29291	
    * F2: 0.25764 
    * Total predictions: 14000 
    * True positives: 477	
    * False positives: 780 
    * False negatives: 1523	
    * True negatives: 11220

* STEP6. Final choise 
  * features_list features_list = ['poi', 'fraction_from_poi', 'fraction_to_poi',
'exercised_stock_options', 'shared_receipt_with_poi']
  * parameters clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=12,max_depth=6, splitter='best' 
  * Metrics Accuracy: 0.86207
  * Precision: 0.52166	
  * Recall: 0.41550	
  * F1: 0.46257	
  * F2: 0.43313 
  * Total predictions: 14000 
  * True positives: 831	
  * False positives: 762 
  * False negatives: 1169	
  * True negatives: 11238
