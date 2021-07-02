import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from joblib import dump, load
from matplotlib import pyplot as plt



data = pd.read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\Natanael_2.csv")


#X = data[["Critical Depth Upper Bound (ft)","Critical Depth Lower Bound (ft)","Critical Depth (ft)","Critical Depth (m)","s critical depth (ft)","Water Depth (ft)","s water table (ft)","sg1 (pcf)","sv (psf)","s sv (psf)","sv' (psf)","s sv' (psf)","rss'","rd"]]

X = data[["Critical Depth Upper Bound (ft)","Critical Depth Lower Bound (ft)","Critical Depth (ft)"]]
y = data["Water Depth (ft)"]

# 65% training, 20%  test, 15% validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.15 / 0.8, random_state = 42)


model = MLPRegressor(hidden_layer_sizes = (2, 18),
                    #random_state = 5,
                    verbose = False,
                    learning_rate_init = 0.01,
                    activation = "tanh",
                    max_iter = 150,
                    early_stopping = False)
    


model.fit(X_train, y_train)



prediction = model.predict(X_test)

print(y_test)
print(prediction)

plt.scatter(y_test, prediction)

line = [max(min(y_test), min(prediction)), min(max(y_test), max(prediction))]

plt.plot(line, line)

plt.xlabel = "measured"
plt.ylabel = "predicted"

plt.show()
