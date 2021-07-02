import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load data
data = pd.read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\HR_comma_sep.csv")


# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers
data["salary"] = le.fit_transform(data["salary"])
data["Departments"] = le.fit_transform(data["Departments"])


X = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "Departments", "salary"]]
y = data["left"]

# 70% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)




# Create full path from relative path
import re
def path(name):
    return re.sub(r'(?<=\\)\w+\.?\w*$',name, __file__, 1)


# Test accuracy from activaion function
def test_activation_function(function):
    clf = MLPClassifier(hidden_layer_sizes = (6, 5),
                    random_state = 5,
                    verbose = True,
                    learning_rate_init = 0.01,
                    activation = function)
    
    print(f"training {function}", end = "")
    
    clf.fit(X_train, y_train)
    model_accuracy = accuracy(clf)
    
    print(f"\t{model_accuracy:.2%} accurate")
    
    return model_accuracy


# Test accuracy of model
def accuracy(model):
    prediction = model.predict(X_test)
    return accuracy_score(y_test, prediction)



functions = ["identity", "logistic", "tanh", "relu"]


result = {}


for function in functions:
  scores = [test_activation_function(function) for i in range(3)]
  best_score = max(scores)
  result[function] = best_score
  print()
  
  
print(result)
#print(__file__)

#print(path(r"models\\model_1.txt"))

                  

