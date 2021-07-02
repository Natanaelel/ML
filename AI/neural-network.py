import math
import random
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt

def read_csv(path):
    file = [*open(path)]
    keys = file[0][3:].rstrip().split(",")
    values = [*map(lambda x: [*map(float, x.split(","))], file[1:])]
#print(keys)
#print(values)

    zipped = [*zip(keys, *values)]
    data = {}

    for column in zipped:
        data[column[0]] = column[1:]

    return data

#data = pd.read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\Natanael_3.csv")

#X = data[["Critical Depth Upper Bound (ft)","Critical Depth Lower Bound (ft)","Critical Depth (ft)"]]
#y = data["Water Depth (ft)"]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#data = read_csv(r"C:\Users\Fredrica Holgersson\Desktop\AI\Natanael_3.csv")

#X = [data["Critical Depth Upper Bound (ft)"],data["Critical Depth Lower Bound (ft)"],data["Critical Depth (ft)"]]
#y = [*map(lambda x: [x], data["Water Depth (ft)"])]

#sets = [*zip([*zip(*X)], y)]

#training_data, testing_data = sets[:140], sets[140:]




class NeuralNetwork:
  learning_rate = 0.5
  def __init__(self, num_in, num_hidden, num_out, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
    
    
    self.num_in = num_in
    self.num_hidden = num_hidden
    self.num_out = num_out
    
    self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
    self.output_layer = NeuronLayer(num_out, hidden_layer_bias)
    
    self.init_weights_ih(hidden_layer_weights )
    self.init_weights_ho(output_layer_weights )
  
  def init_weights_ih(self, hidden_layer_weights):
    weight_num = 0
    for h in range(self.num_hidden):
      for i in range(self.num_in):
        if not hidden_layer_weights:
          self.hidden_layer.neurons[h].weights.append(random.random())
        else:
          self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
        weight_num += 1
    
  
  def init_weights_ho(self, output_layer_weights):
    weight_num = 0
    for o in range(self.num_out):
      for h in range(self.num_hidden):
        if not output_layer_weights:
          self.output_layer.neurons[o].weights.append(random.random())
        else:	
          self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
        weight_num += 1

  
  def feed_forward(self, inputs):
    hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
    return self.output_layer.feed_forward(hidden_layer_outputs)
  
  
  def train(self, training_inputs, training_outputs):
    self.feed_forward(training_inputs)
    
    pd_errors_wrt_output_neuron_total_net_input = [0] * self.num_out
    
    for o in range(self.num_out):
      pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

    pd_errors_wrt_hidden_neuron_total_net_input = [0] * self.num_hidden
    for h in range(self.num_hidden):

      d_error_wrt_hidden_neuron_output = 0
      for o in range(self.num_out):
        d_error_wrt_hidden_neuron_output  += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
      
      pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()
    # update output_layer_weights
    for o in range(self.num_out):
      for w_ho in range(len(self.output_layer.neurons[o].weights)):
        pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
        
        self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * pd_error_wrt_weight
      
    
    
    # update hidden_layer weights
    for h in range(self.num_hidden):
      for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].inputs[w_ih]
        
        self.hidden_layer.neurons[h].weights[w_ih] -= self.learning_rate * pd_error_wrt_weight
        
     
    
  
  
  def calculate_total_error(self, training_sets):
    total_error = 0
    for t in range(len(training_sets)):
      training_inputs, training_outputs = training_sets[t]
      
      self.feed_forward(training_inputs)
      for o in range(len(training_outputs)):
        total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
    
    return total_error    
  


class NeuronLayer:
  def __init__(self, num_neurons, bias):
    self.bias = bias if bias != None else random.random()
    self.neurons = [*map(lambda _: Neuron(self.bias), [0] * num_neurons)]
  
  def feed_forward(self, inputs):
    return [*map(lambda neuron: neuron.calculate_output(inputs), self.neurons)]
  
  def get_outputs(self):
    return [*map(lambda neuron: neuron.output, self.neurons)]
  


class Neuron:
  def __init__(self, bias):
    self.bias = bias
    self.weights = []
    self.inputs = []
    self.output = 0

  
  def calculate_output(self, inputs):
    self.inputs = inputs
    self.output = self.squash(self.calculate_total_net_input())
    return self.output
  

  def calculate_total_net_input(self):
    total = 0
    for i in range(len(self.inputs)):
      total += self.inputs[i] * self.weights[i]
    return total + self.bias
  
  
  def squash(self, net_input):
    return 1 / (1 + math.exp(-net_input))
  
  #def logistic_deriv(self):
  #  return self.output * (1 - this.output)
  #
  
  
  def calculate_pd_error_wrt_total_net_input(self, target_output):
    return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()
  
  
  def calculate_error(self, target_output):
    return 0.5 * (target_output - self.output) ** 2
  
  
  def calculate_pd_error_wrt_output(self, target_output):
    return self.output - target_output
  

  def calculate_pd_total_net_input_wrt_input(self):
    return self.output * (1 - self.output)
  

  def calculate_pd_total_net_input_wrt_weight(self, index):
    return self.inputs[index]


nn = NeuralNetwork(3, 14, 1)

print(nn)

#sets = [
#  [[0, 0], [0]],
#  [[0, 1], [1]],
#  [[1, 0], [1]],
#  [[1, 1], [0]]
#]

sets = [[[20,8,14],[8]],[[16.4,6.4,11.4],[6.4]],[[11.5,1.6,6.5],[1.6]],[[13,3.9,8.5],[3.9]],[[40,12.3,26.2],[12.3]],[[18,3.3,10.7],[3.3]],[[36.1,16.4,26.2],[3]],[[23,11.5,17.2],[3]],[[32.8,16.4,24.6],[3]],[[19.7,6.6,13.1],[2]],[[20,4.5,12.3],[0]],[[20,6.6,13.3],[2]],[[16.4,3,9.7],[2.5]],[[24.6,13.1,18.9],[0]],[[20.7,14.4,17.6],[14]],[[24,17,20.5],[16.3]],[[41,11.5,26.2],[5]],[[29.5,16.4,23],[5]],[[34.4,14.8,24.6],[5]],[[50,10,30],[5]],[[19.7,9.8,14.8],[3.6]],[[19.7,11.5,15.6],[3.3]]]

normalized_sets = preprocessing.normalize([[20,8,14,8],[16.4,6.4,11.4,6.4],[11.5,1.6,6.5,1.6],[13,3.9,8.5,3.9],[40,12.3,26.2,12.3],[18,3.3,10.7,3.3],[36.1,16.4,26.2,3],[23,11.5,17.2,3],[32.8,16.4,24.6,3],[19.7,6.6,13.1,2],[20,4.5,12.3,0],[20,6.6,13.3,2],[16.4,3,9.7,2.5],[24.6,13.1,18.9,0],[20.7,14.4,17.6,14],[24,17,20.5,16.3],[41,11.5,26.2,5],[29.5,16.4,23,5],[34.4,14.8,24.6,5],[50,10,30,5],[19.7,9.8,14.8,3.6],[19.7,11.5,15.6,3.3]])

normalized_sets = [*map(lambda x: [x[:-1], [x[-1]]], normalized_sets)]

print(normalized_sets)
predictions = []
measured = []

for i in range(1000):

  t_input, t_output = random.choice(normalized_sets)
  nn.train(t_input, t_output)
  predicted = nn.feed_forward(t_input)
  predictions.append(predicted)
  measured.append(t_output)
  
  #print(t_input)
  #print(t_output)
  #print(predicted)
  
  if i%10 == 0:
    print(i, nn.calculate_total_error(normalized_sets))


predictied_values = [*map(lambda x: x[0], predictions)]
measured_values = [*map(lambda x: x[0], measured)]

line = [max(min(measured_values), min(predictied_values)), min(max(measured_values), max(predictied_values))]

plt.plot(line, line)

plt.ylabel = "predicted"
plt.xlabel = "measured"

plt.scatter(measured, predictions)

#def step(steps = 1){
#  for i in range steps
#    t_in, t_out = random.choice(sets)
#    nn.train(t_in, t_out)
#    
#  }
#  error = nn.calculate_total_error(sets)
#  #nn.learning_rate = 1/10/error
#  return error
#}





	