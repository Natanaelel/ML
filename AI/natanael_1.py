import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Saving and loading models
from joblib import dump, load


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



# Create model object

clf = MLPClassifier(hidden_layer_sizes = (9, 6),
                    random_state = 5,
                    verbose = True,
                    learning_rate_init = 0.05,
                    activation = "tanh")

# Fit data onto the model
# (Train)
clf.fit(X_train, y_train)

# Make prediction on test dataset
y_prediction = clf.predict(X_test)

#print("\nPrediction:")
#print(" ".join(map(str, y_prediction)))


accuracy = accuracy_score(y_test, y_prediction)




print(f"\nAccuracy: {accuracy:.2%}\t\t{accuracy}")

#res = dump(clf, "model_1.txt")

#print(res)




