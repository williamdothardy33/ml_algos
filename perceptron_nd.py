#note: in the learning algo the binary error measure chosen
#doesn't distinguish between the two ways in which an error
#can be made (y, y') = (+1, -1) (false reject) and (y, y') = (-1, +1) (false accept)
#the same penalty is used but typically this is a domain specific question and should
#be given to you by a person with expertise in the domain that will know how the error
#measure aligns with the business priorities/objectives sometimes there isn't an domain expert
#that can articulate this or the error measure that the domain expert gives you presents 
#challenges in minimization that cannot be overcome with the resources at hand.
#in this case there are a bunch of alternative error measure with very nice 
#analytic properties made by ml/stat/math people that you can choose from but
#these are always second choices
import numpy as np
import matplotlib.pyplot as plt
import math
class perceptron_nd:
    def __init__(self):
        self.rng = np.random.default_rng()

    def cos_sim(self, v_1, v_2):
        dot_product = np.dot(v_1, v_2)
        norm_v_1 = np.linalg.norm(v_1)
        norm_v_2 = np.linalg.norm(v_2)
        return  dot_product / (norm_v_1 * norm_v_2)

    def convergence_bound(self, X, y, target):
        rho = np.min([y[k] * np.dot(target, d) for k, d in enumerate(X)])

        R = np.max([np.linalg.norm(x) for x in X])

        bound = (np.pow(R, 2) * np.pow(np.linalg.norm(target), 2)) / np.pow(rho, 2)

        return bound


    def random_sample_data(self, num_examples, dim,  coord_lb, coord_ub, w):
        X = (coord_ub - coord_lb) * self.rng.random((num_examples, dim)) + coord_lb
        X[:, 0] = 1
        
        y = np.sign(np.dot(X, w))

        return (X, y)
    
    def learning_algo(self, data, num_iter, start):
        w = start

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                y_prime =  np.sign(np.dot(w, X[k]))
                if y_prime != y[k]:
                    w += y[k] * X[k]
                    num_updates += 1
                    finished = False           

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        print(f"it took {i} passes for algorithm to complete\n")

        return w, num_updates

#the error is lagging the hypothesis by one pass need to fix this later
    def pocket_learning_algo(self, data, num_iter, start):
        w = start

        min_error = math.inf
        min_w = w

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            error = 0
            for k in range(num_examples):
                y_prime =  np.sign(np.dot(w, X[k]))
                if y_prime != y[k]:
                    error += 1
                    w += y[k] * X[k]
                    num_updates += 1
                    finished = False
            if error < min_error:
                min_error = error
                min_w = w

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        print(f"in sample error is {min_error}\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        print(f"it took {i} passes for algorithm to complete\n")

        return min_w, num_updates

    def prepared_pocket_learning_algo(self, data, num_iter):

        X, y = data

        num_updates = 0

        w, square_error, _, _ = np.linalg.lstsq(X, y)

        min_error = math.inf
        min_w = w

        print(f"w start is {w}\n")
        print(f"square error for w start is {square_error}\n")


        num_examples = X.shape[0]

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            error = 0
            for k in range(num_examples):
                y_prime =  np.sign(np.dot(w, X[k]))
                if y_prime != y[k]:
                    error += 1
                    w = w + y[k] * X[k]
                    num_updates += 1
                    finished = False
            if error < min_error:
                min_error = error
                min_w = w

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        print(f"in sample error is {min_error}\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        print(f"it took {i} passes for algorithm to complete\n")

        return min_w, num_updates

    def random_update_learning_algo(self, data, num_iter, start):
        w = start

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                y_prime =  np.sign(np.dot(w, X[k]))
                if y_prime != y[k]:
                    finished = False           
                    should_update = self.rng.random()
                    if should_update < 0.5:
                        w += y[k] * X[k]
                        num_updates += 1

        print(f"it took {num_updates} updates for learning algorithm to converge\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        print(f"it took {i} passes for algorithm to complete\n")

        return w, num_updates


    #flip a coin for all inputs instead of just inputs that are misclassified
    #there is a difference theoretically because the "number of trials" is different
    #not sure if computer random number generator makes any guarantees about "emergent" distributions in simulations
    #where in this case I think the random number generator samples from a uniform distribution which is transformed
    #into a bernouli trial by considering half of the interval as "success" and other half as failure
    #and then we do this a bunch of times which should be a binomial distribution I think. don't know how to just get the indices
    #from a binomial distribution that will tell me the position of heads. I can only get the number of heads from the binomial function
    
    def random_input_learning_algo(self, data, num_iter, start):
        w = start

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                should_process = self.rng.random()
                y_prime =  np.sign(np.dot(w, X[k]))
                if should_process < 0.5:
                    if y_prime != y[k]:
                        finished = False
                        w += y[k] * X[k]
                        num_updates += 1
                else:
                    if y_prime != y[k]:
                        finished = False

        print(f"it took {num_updates} updates for learning algorithm to converge\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        print(f"it took {i} passes for algorithm to complete\n")

        return w, num_updates
    

    def adaline_variant_learning_algo(self, data, num_iter, start, update_factor, max_updates):
        w = start
        
        X, y = data

        num_updates = 0

        num_examples = X.shape[0]
        min_w = w
        min_error = np.count_nonzero(np.dot(y, np.dot(X, w)) <= 1) / num_examples

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                if num_updates == max_updates:
                    break
                should_process = self.rng.random()
                signal = np.dot(w, X[k])
                if should_process < 0.5:
                    if y[k] * signal <= 1:
                        finished = False
                        w += update_factor * y[k] * X[k]
                        num_updates += 1
                        error = np.count_nonzero(np.dot(y, np.dot(X, w)) <= 1) / y.size
                        if error < min_error:
                            min_error = error
                            min_w = w
                else:
                    if y[k] * signal <= 1:
                        finished = False

        return min_w, num_updates, min_error
    
    def sim(self, num_examples, coord_lb, coord_ub):
        num_iter = 100000
        dim = 11
        target = (coord_ub - coord_lb) * self.rng.random(dim) + coord_lb
        #start = (coord_ub - coord_lb) * self.rng.random(dim) + coord_lb
        start = np.zeros(dim)

        data = self.random_sample_data(num_examples, dim, coord_lb, coord_ub, target)
        learned, num_updates = self.random_update_learning_algo(data, num_iter, start)

        print(f"target is {target}\n")

        print(f"learned is {learned}\n")


        X, y = data

        convergence_ub = self.convergence_bound(X, y, target)
        print(f"the upperbound for convergence is {convergence_ub}\n")

        #cos_alpha = np.round(self.cos_sim(target[1:], learned[1:]), 3)
        #bias_diff = np.round(np.abs(target[0] - learned[0]), 2)

    def sim_runs(self, num_times):
        num_examples = 1000
        coord_lb = -100
        coord_ub = 100
        dim = 11
        num_iter = 100000
        start = np.zeros(dim)
        target = (coord_ub - coord_lb) * self.rng.random(dim) + coord_lb

        update_freq = np.empty(num_times)

        for i in range(num_times):
            X, y = self.random_sample_data(num_examples, dim, coord_lb, coord_ub, target)
            learned, num_updates = self.random_input_learning_algo((X, y), num_iter, start)
            #print(f"first data point is {X[0]}\n")
            #print(f"learned correctly classified data: {np.all(np.sign(np.dot(X, target)) == np.sign(np.dot(X, learned)))}\n")
            update_freq[i] = num_updates

        plt.hist(update_freq)
        plt.xlabel("number of updates to converge")
        plt.ylabel("frequency")
        plt.title("distribute of number of updates until convergence")
        plt.legend()
        plt.show()

def test():
    num_examples = 1000
    coord_lb = -100
    coord_ub = 100
    p = perceptron_nd()
    p.sim(num_examples, coord_lb, coord_ub)

#test()

def test_runs():
    num_times = 100
    perceptron = perceptron_nd()
    perceptron.sim_runs(num_times)

test_runs()