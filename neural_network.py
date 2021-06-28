import math
import random


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
          self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weights_num])
        weight_num += 1
    
  
  def init_weights_ho(self, output_layer_weights):
    weight_num = 0
    for o in range(self.num_out):
      for h in range(self.num_hidden):
        if not output_layer_weights:
          self.output_layer.neurons[o].weights.append(random.random())
        else:	
          self.output_layer.neurons[o].weights.append(output_layer_weights[weights_num])
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


nn = NeuralNetwork(2, 5, 1)

print(nn)

sets = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]]
]

for i in range(10001):
  t_input, t_output = random.choice(sets)
    
  nn.train(t_input, t_output)
  
  if i%100 == 0:
    print(i, nn.calculate_total_error(sets))



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





	
