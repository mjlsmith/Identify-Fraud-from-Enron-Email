# Enron Submission Free-Response Questions


A critical part of machine learning is making sense of your analysis process and communicating it to others. The questions below will help us understand your decision-making process and allow us to give feedback on your project. Please answer each question; your answers should be about 1-2 paragraphs per question. If you find yourself writing much more than that, take a step back and see if you can simplify your response!


When your evaluator looks at your responses, he or she will use a specific list of rubric items to assess your answers. Here is the link to that rubric: [Link] Each question has one or more specific rubric items associated with it, so before you submit an answer, take a look at that part of the rubric. If your response does not meet expectations for all rubric points, you will be asked to revise and resubmit your project. Make sure that your responses are detailed enough that the evaluator will be able to understand the steps you took and your thought processes as you went through the data analysis.


Once you’ve submitted your responses, your coach will take a look and may ask a few more focused follow-up questions on one or more of your answers.  


We can’t wait to see what you’ve put together for this project!



### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The main goal of this project was to use Machine learning algorithms to investigate persons of interest within the Enron Dataset. The dataset consisted of emails from employees of Enron as well as financial data. The dataset was also broken up to distinguish "POI" (Persons of interest) who, per the project [details](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923), were "individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity."

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

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
