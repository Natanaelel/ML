import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from joblib import dump, load
from matplotlib import pyplot as plt

import time

# Load data
#data = pd.read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\HR_comma_sep.csv")
data = pd.read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\Natanael_2.csv")


# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers
#data["salary"] = le.fit_transform(data["salary"])
#data["Departments"] = le.fit_transform(data["Departments"])


#X = data[["satisfaction_level", "last_evaluation", "number_project", "left", "time_spend_company", "Work_accident", "promotion_last_5years", "Departments", "salary"]]
#y = data["average_montly_hours"]
X = data[["Critical Depth Upper Bound (ft)","Critical Depth Lower Bound (ft)","Critical Depth (ft)","Critical Depth (m)","s critical depth (ft)","Water Depth (ft)","s water table (ft)","sg1 (pcf)","sv (psf)"]]
y = data["Water Depth (ft)"]

# 65% training, 20%  test, 15% validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.15 / 0.8, random_state = 42)

#y_train = [*map(lambda x: round(x/10000), y_train)]
#y_test  = [*map(lambda x: round(x/10000), y_test)]


def save_model(model, name):
    return dump(model, fr"C:\Users\Fredrica Holgersson\Desktop\AI\models\{name}")


# Test accuracy from activaion function
def test_activation_function(function, iterations, save = False, name = "function"):
    clf = MLPClassifier(hidden_layer_sizes = (8, 8),
                    #random_state = 5,
                    verbose = False,
                    learning_rate_init = 0.01,
                    activation = function,
                    max_iter = iterations,
                    early_stopping = True,
                    n_iter_no_change = iterations)
    
    print(f"training {function}")
    
    clf.fit(X_train, y_train)
    
    model_accuracy = accuracy(clf)
    
    print(f"\t{model_accuracy:.2%} accurate")
    
    print(clf.coefs_)
    
    prediction = clf.predict(X_train)
    plt.scatter(y_train, prediction)

    
    if save:
      time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
      save_model(clf, f"{name}_{model_accuracy*100:.2f}_{time_now}.joblib")
    
    return model_accuracy, clf.loss_curve_



# Test accuracy of model
def accuracy(model):
    prediction = model.predict(X_test)
    return accuracy_score(y_test, prediction)



functions = ["identity", "logistic", "tanh", "relu"]
functions = ["tanh", "logistic"]



results = {}

graphs = {}

def graph_functions():
  for function in functions:
    result = [test_activation_function(function, 100, save = False, name = function) for i in range(1)]
  
    best = max(result, key=lambda x: x[0])
  
    results[function] = best[0]
    graphs[function] = best[1]

  
  print(results)



  plt.xlabel("Iterations")
  plt.ylabel("Error")


  for function in graphs:
    skip = 10
  
    errors = graphs[function][skip:]
    
    plt.plot([*range(skip, skip + len(errors))], errors, label = function)
    


  plt.legend()

#graph_functions()



def scatterplot():

    #clf = load(r"C:\Users\Fredrica Holgersson\Desktop\AI\models\relu_model.joblib")
    
    clf = load(r"C:\Users\Fredrica Holgersson\Desktop\AI\models\tanh_1.23_2021-07-01_09-26-14.joblib")
    
    print(clf)
    
    
    split_data = [
        ["train", X_train, y_train],
        ["test", X_test, y_test],
        ["validate", X_validate, y_validate],
    ]
    
    for dataset, X, y in split_data:
        prediction = clf.predict(X)
        #print(*prediction)
        
        plt.scatter(y, prediction, label = dataset)
    
    #plt.xlabel("measured")
    #plt.ylabel("predicted")
    plt.legend()
    
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
    print(clf.score(X_validate, y_validate))





