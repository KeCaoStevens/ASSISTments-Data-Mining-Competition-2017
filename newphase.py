# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:28:21 2017

@author: dongg
"""

from __future__ import print_function
print(__doc__)
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import decomposition
from string import ascii_letters
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from time import time
from operator import itemgetter
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.svm import SVC
data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
X=data.drop(['isSTEM'], 1)
#dropped_X=data.drop(['isSTEM', ], 1)
#names=list(X)


"""
#impute missing mcas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X=input_data
missed_mcas=X.loc[X['MCAS'] == -999]
complete_mcas=X.loc[X['MCAS'] != -999]
mcas=complete_mcas['MCAS']
train_mcas=complete_mcas.drop('MCAS', 1)
selected_train_mcas=complete_mcas.drop(['MCAS', 'responseIsChosen', 'timeSinceSkill', 'totalTimeByPercentCorrectForskill', 'totalFrTimeOnSkill', 'sumTimePerSkill', 'NumActions','frTimeTakenOnScaffolding', 'timeTaken', 'totalFrSkillOpportunities', 'frPast5HelpRequest', 'totalFrAttempted'], 1)
x_train, x_cv, y_train, y_cv = train_test_split(selected_train_mcas,mcas, test_size =0.3)
ridgeReg = Ridge(alpha=0.001, normalize=True)
ridgeReg.fit(x_train,y_train)
pred = ridgeReg.predict(x_cv)
score=ridgeReg.score(x_cv,y_cv)
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)
impute_input = X.drop(X[X.MCAS != -999].index)
impute_input=impute_input.drop(['MCAS','responseIsChosen', 'timeSinceSkill', 'totalTimeByPercentCorrectForskill', 'totalFrTimeOnSkill', 'sumTimePerSkill', 'NumActions','frTimeTakenOnScaffolding', 'timeTaken', 'totalFrSkillOpportunities', 'frPast5HelpRequest', 'totalFrAttempted'], 1)
imputed_mcas=ridgeReg.predict(impute_input)
X.loc[X['MCAS'] == -999, 'MCAS'] = imputed_mcas
newnewdata=pd.concat([X,labels], axis=1)
newnewdata.to_csv('newnewdata.csv', encoding='utf-8', index=False)
"""
"""
#feature selection
ranks={}
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE, f_regression
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
np.random.seed(36)
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))
#random forest
rf = RandomForestClassifier(min_samples_split = 60, 
                             max_leaf_nodes = 10, 
                             n_estimators = 55, 
                             max_depth = 5,
                             min_samples_leaf = 10,
                             min_weight_fraction_leaf= 0.2)
rf.fit(X, y['isSTEM'].values)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

#logistic regression
logReg = LogisticRegression()
logReg.fit(X, y['isSTEM'].values)
ranks["logReg"] = rank_to_dict(np.abs(logReg.coef_[0]), names)

#LDA

lda = LinearDiscriminantAnalysis()
lda.fit(X, y['isSTEM'].values)
ranks["lda"] = rank_to_dict(np.abs(lda.coef_[0]), names)


#support vector machine
svc = svm.SVC(kernel='linear', gamma=1, C=10).fit(X, y['isSTEM'].values)
ranks["svc"] = rank_to_dict(np.abs(svc.coef_[0]), names)


#RFECV
RFECV = RFECV(estimator=logReg, cv=5,
 scoring='mean_squared_error')
RFECV.fit(X, y['isSTEM'].values)
#print ("Features sorted by their rank:")
#ranks["RFECV"] = rank_to_dict(np.abs(selector.ranking_), names)
ranks["RFECV"]=rank_to_dict(RFECV.ranking_, names)

r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print ("\t%s" % "\t".join(methods))
for name in names:
    print ("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
"""
"""
#Standarization
selected_X=X.drop(['AveResConf', 'AveResEngcon', 'AveResGaming', 'AveResOfftask', 'RES_CONCENTRATING', 'RES_CONFUSED', 'RES_FRUSTRATED', 'RES_GAMING', 'RES_OFFTASK', 'consecutiveErrorsInRow', 'endsWithScaffolding', 'frIsHelpRequestScaffolding', 'frPast5WrongCount', 'frPast8WrongCount', 'frTotalSkillOpportunitiesScaffolding', 'frWorkingInSchool', 'hintTotal', 'totalFrSkillOpportunitiesByScaffolding', 'totalFrSkillOpportunities', 'totalFrPercentPastWrong', 'totalFrPastWrongCount', 'timeOver80', 'past8BottomOut'],1)
standarized_X = preprocessing.normalize(selected_X)
name_string = ",".join(list(selected_X ))
np.savetxt('norm.csv', standarized_X, delimiter=',', header=name_string, comments="")
"""

"""
#Random Forest Classifier
#train_ver2, test_ver2, labels=input_data[0:343], input_data[343:], complete_label[0:343].isSTEM.tolist()
train_ver2, test_ver2, labels=input_data[0:343], input_data[343:], complete_label[0:343]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, labels, test_size=0.3, random_state=42)
#grid_times = {}
#clf = RandomForestClassifier(random_state = 84)
#features = X_train
#print (type(y_train))

for number in np.arange(2, 600, 50):
    
    param = np.arange(1,number,10)
    param_grid = {"n_estimators": param,
                  "criterion": ["gini", "entropy"]}
    
    grid_search = GridSearchCV(clf, param_grid = param_grid)
    
    t0 = time()
    grid_search.fit(X_train[features], y_train)
    compute_time = time() - t0
    grid_times[len(grid_search.grid_scores_)] = time() - t0
    
grid_times = pd.DataFrame.from_dict(grid_times, orient = 'index')

# calculate the time to run a GridSearchCV for multiple numbers of parameter permutations.
grid_times = {0: { 2: 0.034411907196044922,
                  12: 1.5366179943084717,
                  22: 5.0431020259857178,
                  32: 11.378448963165283,
                  42: 20.211128950119019,
                  52: 30.040457010269165,
                  62: 39.442277908325195,
                  72: 56.834053993225098,
                  82: 67.847633838653564,
                  92: 91.005517959594727,
                  102: 111.2420859336853,
                  112: 135.75759792327881}}
final = pd.DataFrame.from_dict(grid_times)
final = final.sort_index()
plt.plot(final.index.values, final[0])
plt.xlabel('Number of Parameter Permutations')
plt.ylabel('Time (sec)')
plt.title('Time vs. Number of Parameter Permutations of GridSearchCV')

# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train, y_train['isSTEM'].values)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df

# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(2,150,1),
              "min_samples_leaf": np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):   
    evaluate_param(parameter, param_range, index)
    index += 1


# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# parameters for GridSearchCV
param_grid2 = {"n_estimators": [55, 60, 65, 90, 95, 120, 125],
              "max_depth": [5, 10],
              "min_samples_split": [7, 8, 9, 10, 11, 12],
              "min_samples_leaf": [10,11,12,13,14,15],
              "max_leaf_nodes": [25, 30],
              "min_weight_fraction_leaf": [0.2]}

grid_search = GridSearchCV(clf, param_grid=param_grid2)
grid_search.fit(X_train, y_train['isSTEM'].values)

report(grid_search.grid_scores_, 4)



#clf model
clf = RandomForestClassifier(min_samples_split = 60, 
                             max_leaf_nodes = 10, 
                             n_estimators = 55, 
                             max_depth = 5,
                             min_samples_leaf = 10,
                             min_weight_fraction_leaf= 0.2)
clf.fit(train_ver2, labels['isSTEM'].values)


# importance feature
features = list(input_data)
importances = clf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(15,14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

#rf show
def visualize_tree(tree, feature_names):

    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    
i_tree = 0
for tree_in_forest in clf.estimators_:
    visualize_tree(tree_in_forest, list(input_data))
    i_tree = i_tree + 1

from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
import sklearn
import sklearn.linear_model as lm
import sklearn.grid_search as gs
logreg = lm.LogisticRegression()
selector = RFECV(estimator=logreg, cv=10,
 scoring='mean_squared_error')
selector.fit(X_train, y_train)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), selector.ranking_), list(X_train))))
#X_scaled = selector.transform(X_train)

#SVM

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import GridSearchCV

train_ver2, test_ver2, labels=input_data[0:343], input_data[343:], complete_label[0:343]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, labels, test_size=0.3, random_state=42)
parameters = [{'kernel': ['linear'],
               'gamma': [0.01, 0.1, 1, 10],
                'C': [10, 100, 1000]}
              ]
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train['isSTEM'].values)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
"""

#MLP
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.layers.core import Dropout, Activation
#from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
#from keras.constraints import max_norm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
"""
"""
data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
y=pd.DataFrame.as_matrix(y)
y=y.reshape((492, ))
#y_bool=y.astype('bool')
import itertools
from sklearn import svm
#X=pd.read_csv('norm.csv')
X=pd.DataFrame.as_matrix(X)
stand_X=pd.read_csv('standarized_X.csv')
#print (type(X))
"""
def create_model(init_mode='normal', activation='relu',dropout_rate=0.1, neurons=70):
    model = Sequential()
    model.add(Dense(35, input_dim=35,init=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, init=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer =SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
seed = 7
np.random.seed(seed)

#model = KerasClassifier(build_fn=create_model, verbose=0)

#batch_size = [10]
#epochs = [5]
"""
"""
init_mode = ['normal']
activation = ['relu']
dropout_rate = [0.01]
neurons = [70]
optimizer =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=10, verbose=0)
param_grid = dict(neurons=neurons, dropout_rate=dropout_rate, init_mode=init_mode, activation=activation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-11, cv=10)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%0.3f (+/-%0.03f) for %r" % (mean, stdev, param))
"""
"""

mlp=MLPClassifier()
param_grid_linear = {
        "activation": ['relu', 'logistic', 'tanh'],
        "hidden_layer_sizes": [(35, 80, 1)],
        "solver": ['sgd', 'adam'], 
        "learning_rate": ['invscaling', 'adaptive', 'constant'],
        "max_iter": [800, 1000],
        "momentum": [0.3, 0.5, 0.8],
        }

X_train, X_test, y_train, y_test = train_test_split(stand_X, y, test_size=0.30, random_state=7)
CV_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_linear, cv= 10)

CV_mlp.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(CV_mlp.best_params_)
print()
print("Grid scores on training set:")
print()
means = CV_mlp.cv_results_['mean_test_score']
stds = CV_mlp.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, CV_mlp.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
"""


#X_train, X_test, y_train, y_test = train_test_split(stand_X[0:350], y[0:350], test_size=0.30, random_state=7)
mlp = MLPClassifier(activation='relu', solver = 'sgd', learning_rate = 'adaptive', max_iter = 1000, hidden_layer_sizes=(35,70,1))
mlp.fit(stand_X[0:392], y[0:392])
predictions = mlp.predict(stand_X[392:])
print(confusion_matrix(y[392:],predictions))
print (classification_report(y[392:],predictions))
print (accuracy_score(y[392:], predictions))
"""
def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()


print ("plotting")
mlp.fit(X, y)
plot_coefficients(mlp, list(X))
"""


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix
cnf_matrix = confusion_matrix(y[392:], predictions)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print ("Accuracy Score:")
print (accuracy_score(y[392:], predictions))


"""
svc = SVC()
param_grid_linear = { 
           "kernel" : ['rbf'],
           "gamma" : [1, 10, 0.1, 100],
           "C" : [10, 1, 0.1],
            }
X__cvtrain, X_cvtest, y_cvtrain, y_cvtest = train_test_split(
    X, y, test_size=0.30, random_state=10)
CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid_linear, cv= 10)

CV_svc.fit(X__cvtrain, y_cvtrain)
print("Best parameters set found on development set:")
print()
print(CV_svc.best_params_)
print()
print("Grid scores on training set:")
print()
means = CV_svc.cv_results_['mean_test_score']
stds = CV_svc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, CV_svc.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
#print()
y_pred = svc.fit(X__cvtrain, y_cvtrain).predict(X_cvtest)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix
cnf_matrix = confusion_matrix(y_cvtest, y_pred)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print ("Accuracy Score:")
print (accuracy_score(y_cvtest, y_pred))


def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()


print ("plotting")
svc.fit(X, y)
plot_coefficients(svc, list(X))

clf = RandomForestClassifier(min_samples_split = 60, 
                             max_leaf_nodes = 10, 
                             n_estimators = 55, 
                             max_depth = 5,
                             min_samples_leaf = 10,
                             min_weight_fraction_leaf= 0.2)
clf.fit(X_train, y_train)
y_pred = clf.fit(X_train, y_train).predict(X)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print (accuracy_score(y_test, y_pred))

# importance feature
features = list(X)
importances = clf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(15,14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()
"""