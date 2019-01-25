# EEG_Classification
Age-related classification of EEG sessions.

The most significant features of the EEG sessions are identified using two-sample (Student's) t-test. These significant features are used to train different classifiers to distinguish the EEG session as belonging to younger or older adult groups.

This repository has code to perform two-sample t-test and, to train the following classification methods: 
* Least Squares Linear classifier
* K-nearest neighbours classifier
* Extreme Learning Machine (ELM) with linear, sigmoid and RBF kernels
* Support Vector Machine (SVM) with linear and RBF kernels

For each of these methods, the accuracy, confusion matrix and area under (ROC) curve (AUC) scores are calculated.

Also, a script is available to detect outliers in a class.
 
## Dependencies

Programming Language: Python 3.6

To be able to run this code, the following modules need to be installed:
* NumPy
* SciPy
* scikit-learn
* sklearn-extensions (https://pypi.org/project/sklearn-extensions/)

## How to run the code?

The main script identifies the most significant features, performs different classification methods on the data and reports the accuracy, confusion matrix and area under (ROC) curve (AUC) scores for each classifier.

```
python main.py --class1 data_class1.mat --class2 data_class2.mat --nSessions 2 [--field_name data]

    --class1     - a .mat file containing the data of class1
    --class2     - a .mat file containing the data of class2
    --nSessions  - number of sessions per subject
    --field_name - name of the field under which the data is available in the .mat file (optional)
```
** Note: The field name of the data in the dictionary of .mat file should match the (string) value of `field_name`.


The following script detects the outliers in both the classes. It reports the outliers identified by Local Outlier Factor and Isolation Forest methods.

```
python detect_outliers.py --class1 data_class1.mat --class2 data_class2.mat [--field_name data]

    --class1     - a .mat file containing the data of class1
    --class2     - a .mat file containing the data of class2
    --nSessions  - number of sessions per subject
    --field_name - name of the field under which the data is available in the .mat file (optional)
```
** Note: The field name of the data in the dictionary of .mat file should match the (string) value of `field_name`.

## Reference
