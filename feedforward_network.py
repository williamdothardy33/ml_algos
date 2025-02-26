import numpy as np
import math
class feedforward_network:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.w = None
        self.w_gradient = None
        self.x_outs = None
        self.point_errors = None
        self.previous_batch_error = None

    def init_w_gradient(self):
        if not self.w_gradient:
            print('warning network needs architecture configuration before running <init_w_gradient>\n')
            print('please run <init_network_from> with a valid architecture\n')
            return

        for w in self.w_gradient:
            w[::] = 0

    def init_point_errors(self, num_examples):
        if not self.w_gradient:
            print('warning network needs architecture configuration before running <init_point_errors>\n')
            print('please run <init_network_from> with a valid architecture\n')
            return

        network_out_size = self.w_gradient[-1].shape[1]
        self.point_errors = np.zeros((num_examples, network_out_size))

        



    def init_network_from(self, architecture, max_square_input, scale_factor):
        self.w = [self.rng.normal(0, scale_factor * (1 / max_square_input), (architecture[num_nodes] + 1, architecture[num_nodes + 1])) for num_nodes in range(architecture.size - 1)]
        self.w_gradient = [np.zeros_like(self.w[l]) for l in range(len(self.w))]
        self.x_outs = [np.ones(architecture[n] + 1) if n < architecture.size - 1 else np.zeros(architecture[n]) for n in range(architecture.size)]


    def tanh_threshold(self, s_in, layer):
        np.tanh(s_in, out=self.x_outs[layer][1:])
    
    def tanh_threshold_rate(self, layer):
        x = self.x_outs[layer]
        return 1 -  (x ** 2)[1:]

    def tanh_prediction_threshold(self, s_in):
        np.tanh(s_in, out=self.x_outs[-1])



    def forward(self, init_input):
        layers = len(self.w_gradient)
        self.x_outs[0][1:] = init_input
        for layer in range(layers - 1):
            self.tanh_threshold(np.dot(np.transpose(self.w[layer]), self.x_outs[layer]), layer + 1)
        
        self.tanh_prediction_threshold(np.dot(np.transpose(self.w[-1]), self.x_outs[-2])) 
        

    def backward_step(self, w_out_layer, current_sensitivity):
        return self.tanh_threshold_rate(w_out_layer) * np.dot(self.w[w_out_layer][1:,], current_sensitivity)

    def point_gradient_update(self, x, y, num_examples):
        self.forward(x)

        current_sensitivity = 2 * (self.x_outs[-1] - y) * (1 - self.x_outs[-1] ** 2)
        previous_sensitivity = current_sensitivity

        w_out_layers = len(self.w_gradient) - 1

        for w_out_layer in range(w_out_layers, 0, -1):
            current_sensitivity = self.backward_step(w_out_layer, current_sensitivity)
            self.w_gradient[w_out_layer] += np.outer((1 / num_examples) * self.x_outs[w_out_layer], previous_sensitivity)
            previous_sensitivity = current_sensitivity

        self.w_gradient[0] += np.outer((1 / num_examples) * self.x_outs[0], previous_sensitivity)

    def point_square_error(self, predicted, actual):
        return (predicted - actual) ** 2

    def batch_learning_algo(self, training_data, descent_rate, max_epochs, should_stop, min_error):
        X, Y = training_data
        #need to figure out when to stop
        num_examples = X.shape[0]

        for _ in range(max_epochs):
            self.init_w_gradient()
            for i in range(num_examples):
                self.point_gradient_update(X[i], Y[i], num_examples)
                self.point_errors[i] = self.point_square_error(self.x_outs[-1], Y[i])

            for l, delta_w in enumerate(self.w_gradient):
                self.w[l] += -descent_rate * delta_w

            #probably will need to run forward to get new predictions after the last back_prop
            #the errors will lag behind by 1 epoch I think. need to figure out how to get realtime error efficiently
            self.previous_batch_error = np.mean(self.point_errors)

    def stochastic_learning_algo(self, training_data, descent_rate, max_epochs, should_stop, min_error):
        X, Y = training_data
        #need to figure out when to stop
        num_examples = X.shape[0]

        for _ in range(max_epochs):
            for i in self.rng.choice(num_examples, num_examples, replace=False):
                self.init_w_gradient()

                self.point_gradient_update(X[i], Y[i], 1)

                for l, delta_w in enumerate(self.w_gradient):
                    self.w[l] += -descent_rate * delta_w
                    
                self.point_errors[i] = self.point_square_error(self.x_outs[-1], Y[i])

            

            #probably will need to run forward to get new predictions after the last back_prop
            #the errors will lag behind by 1 epoch I think. need to figure out how to get realtime error efficiently
            #ok guiding principle for stopping criteria:
            #1) detect when you are at a local minimum
            #- could possibly do this by keeping a window average of size of change in weight vector
            #- this will let me know by proxy marginal change in error and if error surface is getting flatter
            #- could possibly start tracking weight vector when I detect I'm at local minimum and keep the best one
            # like the pocket algorithm does
            #- possibly could explore multiple minimum or get in and jump out of one never to find another one
            #2) compute real time error if at local minimum and check whether it meets tolerance
            #3) if it doesn't check and see if there are resources available for longer computation
            #4) later on may add some code to stop based on test set error or something which I suspect is what validation is
            #5) will add regularization and other goodies after I learn about them
            
            self.previous_batch_error = np.mean(self.point_errors)
            


            

        






def test_point_gradient_update():
    x = np.array([2])
    y = np.array([1])
    nn = feedforward_network()
    W_1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    W_2 = np.array([[0.2], [1.0], [-3.0]])
    W_3 = np.array([[1.0], [2.0]])
    w = [W_1, W_2, W_3]
    architecture = np.array([1, 2, 1, 1])
    #nn.x_outs = [np.ones(architecture[n] + 1) if n < architecture.size - 1 else np.zeros(architecture[n]) for n in range(architecture.size)]
    nn.init_network_from(architecture, np.dot(x, x), 0.5)
    print(f'x outs is initially {nn.x_outs}\n')
    print(f'after running <init_network_from> w_gradient is {nn.w_gradient}\n')


    nn.w = w

    print(f'w is initially {nn.w}\n')

    nn.point_gradient_update(x, y, 1)

    print(f'after calculating point gradient x outs is {nn.x_outs}\n')
    print(f'after calculating point gradient w_gradient is {nn.w_gradient}\n')

    nn.init_w_gradient()

    print(f'after running <init_w_gradient> w_gradient is {nn.w_gradient}\n')

test_point_gradient_update()



    
    
