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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.15 / 0.8, random_state = 42)



def save_model(model, name):
    return dump(model, fr"C:\Users\Fredrica Holgersson\Desktop\AI\models\{name}")


# Test accuracy from activaion function
def test_activation_function(function, iterations):
    clf = MLPClassifier(hidden_layer_sizes = (6, 5),
                    random_state = 5,
                    verbose = False,
                    learning_rate_init = 0.01,
                    activation = function,
                    max_iter = iterations,
                    early_stopping = False,
                    n_iter_no_change = 10000)
    
    print(f"training {function}")
    
    clf.fit(X_train, y_train)
    model_accuracy = accuracy(clf)
    
    print(f"\t{model_accuracy:.2%} accurate")
    
    return model_accuracy, clf.loss_curve_


# Test accuracy of model
def accuracy(model):
    prediction = model.predict(X_test)
    return accuracy_score(y_test, prediction)



functions = ["identity", "logistic", "tanh", "relu"]


result = {}

graphs = {}


for function in functions:
  results = [test_activation_function(function, 100) for i in range(1)]
  
  best = max(results, key=lambda x: x[0])
  
  result[function] = best[0]
  graphs[function] = best[1]
  
  print()
  
plt.xlabel("Iterations")
plt.ylabel("Error")


for function in graphs:
  skip = 10
  
  errors = graphs[function][skip::]
    
  plt.plot([*range(skip, skip + len(errors))], errors, label = function)
    


plt.legend()

