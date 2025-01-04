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
#note: I need to fix random input versions so that inputs a chosen uniformly
import numpy as np
import matplotlib.pyplot as plt
import math

class perceptron_2d:
    def __init__(self, seed = None):
        if not seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

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

    def predicted_probability(self, w, X):
        return 1 / (1 + np.exp(-np.dot(X, w))) 
    
    def binary_cross_entropy(self, data, w):
        X, y = data
        return np.mean(np.log(1 + np.exp(-y * np.dot(X, w))))
    
    def random_binary_cross_entropy(self, data, w):
        X, y = data
        i = self.rng.choice(X.shape[0])

        return np.log(1 + np.exp(-y[i] * np.dot(w, X[i])))

    
    def error_descent_bearing(self, data, w_t):
        X, y = data
        return -np.mean((np.reshape(y, (-1, 1)) * X) / np.reshape((1 + np.exp(y * np.dot(X, w_t))), (-1,1)), axis=0)

    #expected value of the "scenic" direction is the batch direction
    def random_error_descent_bearing(self, data, w_t):
        X, y = data
        i = self.rng.choice(X.shape[0])

        return -(y[i] * X[i]) / (1 + np.exp(y[i] * np.dot(w_t, X[i])))
    
    def pointwise_error_descent_bearing(self, data, w_t):
        x, y = data

        return -(y * x) / (1 + np.exp(y * np.dot(w_t, x)))

    def logistic_learning_algo(self, training_data, start, descent_rate, max_epochs, min_error, min_delta_error):
        w_t = start.copy()
        X, y = training_data
        current_error = self.binary_cross_entropy(training_data, w_t)
        previous_error = current_error
        
        for i in range(max_epochs):
            if current_error <= min_error:
                break

            next_bearing = -self.error_descent_bearing(training_data, w_t)
            w_t += descent_rate * next_bearing

            previous_error = current_error
            current_error = self.binary_cross_entropy(training_data, w_t)

            if np.linalg.norm(next_bearing) <= min_delta_error and ((max_epochs - i) / max_epochs) < descent_rate:
                break

        return w_t, current_error
    

    def logistic_sgd_learning_algo(self, training_data, start, descent_rate, max_epochs, min_error):
        previous_w_t = self.rng.random(3)
        w_t = start.copy()
        X, y = training_data
        #print(f'initial change in weight vector is {np.sqrt(np.dot(previous_w_t - w_t, previous_w_t - w_t))}\n')
        num_examples = X.shape[0]
        current_error = self.binary_cross_entropy(training_data, w_t)
        #print(f'error started at {current_error}\n')
        finished = False
        for i in range(max_epochs):
            if finished:
                break
            previous_w_t = w_t.copy()

            for k in self.rng.choice(num_examples, num_examples, replace=False):
                if current_error <= min_error:
                    finished = True
                    break

                next_bearing = -self.pointwise_error_descent_bearing((X[k], y[k]), w_t)
                w_t += descent_rate * next_bearing
                current_error = self.binary_cross_entropy(training_data, w_t)
                #print(f'after update error is: {current_error}\n')

            if np.linalg.norm((previous_w_t - w_t)) < 0.01:
                #print(f'change in weight vector has length: {np.linalg.norm((previous_w_t - w_t))}\n')
                #print(f'change in weight vector calculate by hand after update  is {np.sqrt(np.dot(previous_w_t - w_t, previous_w_t - w_t))}\n')
                print(f'the predicted probability for the positive class is\n{np.round(self.predicted_probability(w_t, X), 2)}\n')
                print(f'the actual labels are {y}\n')
                finished = True
                break

        #print(f'the current error after terminating is {current_error}\n')
        return w_t, current_error, i




    def random_sample_data(self, num_examples,  coord_lb, coord_ub, w):
        X = self.rng.uniform(coord_lb, coord_ub, (num_examples, 3))
        #X = (coord_ub - coord_lb) * self.rng.random((num_examples, 3)) + coord_lb
        X[:, 0] = 1
        
        y = np.sign(np.dot(X, w))

        return (X, y)
    
    def random_sample_data_nonlinear(self, num_examples,  coord_lb, coord_ub, f):
        X = self.rng.uniform(coord_lb, coord_ub, (num_examples, 3))
        X[:, 0] = 1
        
        y = f(X)

        return (X, y)
    
    def make_noise(self, outputs, rate):
        indices = self.rng.choice(outputs.size, int(np.ceil(rate * outputs.size)), False)
        result = outputs.copy()
        for idx in indices:
            result[idx] *= -1 

        return result
    
    def learning_algo(self, data, num_iter, start):
        w = start.copy()

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

        return w, num_updates

    def pocket_learning_algo(self, data, num_iter, start):
        w = start.copy()

        min_w = w

        X, y = data

        num_examples = X.shape[0]

        min_error = np.count_nonzero((np.dot(X, w)) != y) / num_examples

        num_updates = 0

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
                    error = np.count_nonzero((y * np.dot(X, w)) <= 0) / num_examples
                    if error < min_error:
                        min_error = error
                        min_w = w

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        print(f"in sample error is {min_error}\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        return min_w, num_updates
    
    #I didn't fully pin down the reasoning when I first started randomizing the inputs but intuitively if we are going to stop after 
    # some predetermined number of updates it is best to make sure the weight space is searched as evenly as possible so I think the pocket 
    # algorithm makes the most since when inputs are evaluated randomly
    
    def random_pocket_learning_algo(self, data, num_iter, start, max_updates = None, diagnostic = False, suppress_msg = True):
        w = start.copy()

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        min_w = w.copy()

        min_error = np.mean(np.sign(np.dot(X, w)) != y)

        if max_updates:
            ws = np.empty((max_updates + 1, 3))
            min_ws = np.empty((max_updates + 1, 3))
            min_ws[0] = start.copy()
            ws[0] = start.copy()
            offset = 1




        if not max_updates:
            max_updates = math.inf

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                if num_updates == max_updates:
                    finished = True
                    break

                should_process = self.rng.random()
                signal = np.dot(w, X[k])
                if should_process < 0.5:
                    if y[k] != np.sign(signal):
                        finished = False
                        w += y[k] * X[k]
                        num_updates += 1
                        error = np.mean(np.sign(np.dot(X, w)) != y)
                        if error < min_error:
                            min_error = error
                            min_w = w.copy()

                        if diagnostic:
                            ws[offset] = w.copy()
                            min_ws[offset] = min_w.copy()
                            offset += 1

                else:
                    if y[k] != np.sign(signal):
                        finished = False
        if not suppress_msg:
            if num_updates == max_updates:
                print(f"algorithm may not have converged within {max_updates} updates\n")

            if i == num_iter:
                print(f"algorithm may not have completed in {num_iter} iterations\n")

        if diagnostic:
            if not suppress_msg:
                print(f"hypothesis is {min_w}\n")
                print(f"it took {num_updates} updates for learning algorithm to converge\n")
                print(f"in sample error is {min_error}\n")

            return min_ws, ws

        

        return min_w, num_updates, min_error
    
    def uniform_pocket_learning_algo(self, data, max_epochs, start, max_updates = None, diagnostic = False, suppress_msg = True):
        w = start.copy()

        X, y = data

        num_updates = 0

        num_examples = X.shape[0]

        min_w = w.copy()

        min_error = np.mean(np.sign(np.dot(X, w)) != y)

        if max_updates:
            ws = np.empty((max_updates + 1, 3))
            min_ws = np.empty((max_updates + 1, 3))
            min_ws[0] = start.copy()
            ws[0] = start.copy()
            offset = 1




        if not max_updates:
            max_updates = math.inf

        finished = False
        for i in range(max_epochs):
            if finished:
                break
            finished = True
            for k in self.rng.choice(num_examples, num_examples, replace=False):
                if num_updates == max_updates:
                    finished = True
                    break

                signal = np.dot(w, X[k])
                if y[k] != np.sign(signal):
                    finished = False
                    w += y[k] * X[k]
                    num_updates += 1
                    error = np.mean(np.sign(np.dot(X, w)) != y)
                    if error < min_error:
                        min_error = error
                        min_w = w.copy()

                    if diagnostic:
                        ws[offset] = w.copy()
                        min_ws[offset] = min_w.copy()
                        offset += 1

        if not suppress_msg:
            if num_updates == max_updates:
                print(f"algorithm may not have converged within {max_updates} updates\n")

            if i == max_epochs:
                print(f"algorithm may not have completed in {max_epochs} epochs\n")

        if diagnostic:
            if not suppress_msg:
                print(f"hypothesis is {min_w}\n")
                print(f"it took {num_updates} updates for learning algorithm to converge\n")
                print(f"in sample error is {min_error}\n")

            return min_ws, ws

        

        return min_w, num_updates, min_error

    def prepared_pocket_learning_algo(self, data, num_iter):

        X, y = data

        num_examples = X.shape[0]

        num_updates = 0

        w, square_error, _, _ = np.linalg.lstsq(X, y)

        min_error = np.count_nonzero((y * np.dot(X, w)) <= 0) / num_examples

        min_w = w

        print(f"w start is {w}\n")
        print(f"square error for w start is {square_error}\n")

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                y_prime =  np.sign(np.dot(w, X[k]))
                if y_prime != y[k]:
                    w = w + y[k] * X[k]
                    num_updates += 1
                    finished = False
                    error = np.count_nonzero((y * np.dot(X, w)) <= 0) / num_examples
                    if error < min_error:
                        min_error = error
                        min_w = w

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        print(f"in sample error is {min_error}\n")

        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        return min_w, num_updates
    
    def ls_learning_algo(self, data, diagnostic = False):

        X, y = data

        num_examples = X.shape[0]

        w, square_error, _, _ = np.linalg.lstsq(X, y)

        error = np.count_nonzero((y != np.sign(np.dot(X, w)))) / num_examples

        
        if diagnostic:
            print(f"w is {w}\n")
            print(f"square error for w is {square_error}\n")

        return w, error
    
    #waaaaaay too slow goddamn
    
    # def random_update_learning_algo(self, data, num_iter, start):
    #     w = start

    #     X, y = data

    #     num_updates = 0

    #     num_examples = X.shape[0]

    #     finished = False
    #     for i in range(num_iter):
    #         if finished:
    #             break
    #         finished = True
    #         mislabeled_indices = np.full(y.size, -1)
    #         in_offset = 0
    #         count = 0

    #         for k in range(num_examples):
    #             y_prime =  np.sign(np.dot(w, X[k]))
    #             if y_prime != y[k]:
    #                 mislabeled_indices[in_offset] = k
    #                 in_offset += 1
    #                 count += 1
    #                 #w += y[k] * X[k]
    #                 #num_updates += 1
    #                 finished = False

    #         if count != 0:
    #             update_index = self.rng.choice(count)
    #             w += y[mislabeled_indices[update_index]] * X[mislabeled_indices[update_index]]
    #             num_updates += 1

    #     print(f"it took {num_updates} updates for learning algorithm to converge\n")
    #     if i == num_iter:
    #         print(f"algorithm may not have completed in {num_iter} iterations\n")

    #     return w, num_updates

    def random_update_learning_algo(self, data, num_iter, start):
        w = start.copy()

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
                    should_update = self.rng.choice(2)
                    if should_update == 1:
                        w += y[k] * X[k]
                        num_updates += 1

        print(f"it took {num_updates} updates for learning algorithm to converge\n")
        if i == num_iter:
            print(f"algorithm may not have completed in {num_iter} iterations\n")

        return w, num_updates
    

    def adaline_variant_learning_algo(self, data, num_iter, start, update_factor, max_updates):
        w = start.copy()
    
        X, y = data

        num_updates = 0

        num_examples = X.shape[0]
        min_w = w
        #min_error = np.count_nonzero((y * np.dot(X, w)) <= 1) / num_examples
        min_error = np.count_nonzero((y != np.sign(np.dot(X, w)))) / num_examples

        finished = False
        for i in range(num_iter):
            if finished:
                break
            finished = True
            for k in range(num_examples):
                if num_updates == max_updates:
                    finished = True
                    break
                should_process = self.rng.random()
                signal = np.dot(w, X[k])
                if should_process < 0.5:
                    #if y[k] * signal <= 1:
                    if y[k] != np.sign(signal):
                        finished = False
                        w += update_factor * y[k] * X[k]
                        #w += update_factor * (y[k] - signal) * X[k]

                        num_updates += 1
                        #error = np.count_nonzero(np.dot(y, np.dot(X, w)) <= 1) / num_examples
                        error = np.count_nonzero((y != np.sign(np.dot(X, w)))) / num_examples
                        if error < min_error:
                            min_error = error
                            min_w = w
                else:
                    #if y[k] * signal <= 1:
                    if y[k] !=  np.sign(signal):

                        finished = False

        return min_w, num_updates, min_error
    
    def sim_2d(self, num_examples, coord_lb, coord_ub):
        num_iter = 100000
        #target = (coord_ub - coord_lb) * self.rng.random(3) + coord_lb
        target = self.rng.uniform(coord_lb, coord_ub, 3)
        #start = (coord_ub - coord_lb) * self.rng.random(3) + coord_lb
        start = np.zeros(3)
        data = self.random_sample_data(num_examples, coord_lb, coord_ub, target)
        learned, num_updates = self.random_update_learning_algo(data, num_iter, start)

        print(f"target is {target}\n")
        print(f"weights for target are {target[1:]}")
        print(f"slope of target is {-(target[1] / target[2])}\n")

        print(f"learned is {learned}\n")
        print(f"weights for learned are {learned[1:]}")
        print(f"slope of learned is {-(learned[1] / learned[2])}\n")


        X, y = data

        convergence_ub = self.convergence_bound(X, y, target)
        print(f"the upperbound for convergence is {convergence_ub}\n")


        _, ax = plt.subplots()
        
        for classification in [-1, 1]:
            mask = (y == classification)
            ax.scatter(X[:, 1][mask], X[:, 2][mask], marker= 'x' if classification ==  -1 else 'o', label='-1' if classification ==  -1 else '+1')

        x = np.arange(coord_lb - 1, coord_ub + 1, 1)

        target_y = -(target[1] / target[2]) * x  - (target[0] / target[2])

        learned_y = -(learned[1] / learned[2]) * x  - (learned[0] / learned[2])
        #cos_alpha = np.round(self.cos_sim(target[1:], learned[1:]), 3)
        #bias_diff = np.round(np.abs(target[0] - learned[0]), 2)

        plt.plot(x, target_y, label="target boundary")
        plt.plot(x, learned_y, label="learned boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"perceptron sim\nnumber of examples={num_examples}\nnumber of updates={num_updates}\nupperbound for convergence={convergence_ub}\n")
        plt.legend()
        plt.show()

    def sim_adaline(self):
        num_iter = 100000
        num_training_examples = 100
        num_test_examples = 10000
        coord_lb = -2000
        coord_ub = 2000
        max_updates = 1000
        max_update_factor = 100
        #target = (coord_ub - coord_lb) * self.rng.random(3) + coord_lb
        target = self.rng.uniform(coord_lb, coord_ub, 3)
        start = np.zeros(3)

        test_data = self.random_sample_data(num_test_examples, coord_lb, coord_ub, target)
        training_data = self.random_sample_data(num_training_examples, coord_lb, coord_ub, target)


        update_factors = np.array([max_update_factor / (100 ** i) for i in range(4)])

        learned = np.empty((update_factors.size, 3))
        num_updates = np.empty(update_factors.size)
        in_sample_errors = np.empty(update_factors.size)
        out_sample_errors = np.empty(update_factors.size)


        for k, update_factor in enumerate(update_factors):
            learned[k], num_updates[k], in_sample_errors[k] = self.adaline_variant_learning_algo(training_data, num_iter, start, update_factor, max_updates)


        X_test, y_test = test_data

        for j, h in enumerate(learned):
            out_sample_errors[j] = np.count_nonzero(y_test !=  np.sign(np.dot(X_test, h))) / num_test_examples

        
        X_train, y_train = training_data

        _, axs = plt.subplots(1, update_factors.size, tight_layout=True)


        x = np.arange(coord_lb - 1, coord_ub + 1, 1)
        for u, hypothesis in enumerate(learned):
        
            for classification in [-1, 1]:
                mask = (y_train == classification)
                axs[u].scatter(X_train[:, 1][mask], X_train[:, 2][mask], marker= 'x' if classification ==  -1 else 'o', label='-1' if classification ==  -1 else '+1')

                target_y = -(target[1] / target[2]) * x  - (target[0] / target[2])
                learned_y = -(hypothesis[1] / hypothesis[2]) * x  - (hypothesis[0] / hypothesis[2])

                axs[u].plot(x, target_y, label="target boundary")
                axs[u].plot(x, learned_y, label="learned boundary")
                axs[u].set_title(f"adaline sim\nupdate factor={update_factors[u]}\nnumber of examples={num_training_examples}\nnumber of updates={num_updates[u]}\nin sample error={in_sample_errors[u]}\nout of sample errors={out_sample_errors[u]}\n")
                axs[u].set_xlabel("Feature 1")
                axs[u].set_ylabel("Feature 2")
        
        plt.legend()
        plt.show()

    def sample_custom_clamp_target(self, num_examples, rad, thk, sep):
        t = self.rng.uniform(0, 2 * np.pi, num_examples)
        r = self.rng.uniform(rad, rad + thk, num_examples)

        x_offset = rad + (thk / 2)

        y_offset = -sep

        One = np.ones((num_examples, 2))

        R = np.transpose(np.array([np.cos(t), np.sin(t)]))

        C = np.transpose(np.array([x_offset * np.ones(num_examples), y_offset * np.ones(num_examples)]))

        positive_mask = (0 <= t) * (t < np.pi)

        y_train = np.where(positive_mask, 1, -1)
        X_train = np.where(np.transpose(np.array([positive_mask, positive_mask])), r.reshape(-1, 1) * One * R, C + (r.reshape(-1, 1) * One * R))

        
        #negative_mask = positive_mask != True


        # t_pos = t[positive_mask]

        # t_neg = t[negative_mask]

        # r_pos = r[positive_mask]

        # r_neg = r[negative_mask]

        # pos_class_x = r_pos * np.ones_like(t_pos) * np.cos(t_pos)
        # pos_class_y = r_pos * np.ones_like(t_pos) * np.sin(t_pos)

        # neg_class_x = x_offset + r_neg * np.ones_like(t_neg) * np.cos(t_neg)
        # neg_class_y = -sep + r_neg * np.ones_like(t_neg) * np.sin(t_neg)

        # return (pos_class_x, pos_class_y, neg_class_x, neg_class_y)

        return X_train, y_train


def test():
    num_examples = 1000
    coord_lb = -100
    coord_ub = 100
    p = perceptron_2d()
    p.sim_2d(num_examples, coord_lb, coord_ub)

#test()

def test_adaline():
    p = perceptron_2d()
    p.sim_adaline()
#test_adaline()

def test_hw1p7():
    p = perceptron_2d()
    num_iter = 100000
    num_runs = 1000
    #num_training_examples = 10
    num_training_examples = 100
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    max_updates = 1000
    update_factor = 1
    
    target = p.rng.uniform(coord_lb, coord_ub, 3)
    start = np.zeros(3)
    training_data = p.random_sample_data(num_training_examples, coord_lb, coord_ub, target)
    X_test, y_test = p.random_sample_data(num_test_examples, coord_lb, coord_ub, target)

    X_train, y_train = training_data

    # for k, output in enumerate(y_train):
    #     input = X_train[k]
    #     if output == 0:
    #         while output == 0:
    #             input = p.rng.uniform(coord_lb, coord_ub, 3)
    #             input[0] = 1
    #             output = np.sign(np.dot(target, input))
    #         X_train[k] = input
    #         y_train[k] = output


    
            



    learned = np.empty((num_runs, 3))
    num_updates = np.empty(num_runs)
    in_sample_errors = np.empty(num_runs)
    out_sample_errors = np.empty(num_runs)
    
    for run_i in range(num_runs):
        learned[run_i], num_updates[run_i], in_sample_errors[run_i] = p.adaline_variant_learning_algo(training_data, num_iter, start, update_factor, max_updates)
        out_sample_errors[run_i] = np.count_nonzero(y_test !=  np.sign(np.dot(X_test, learned[run_i]))) / num_test_examples



    print(f"the average number of updates before convergence is {np.average(num_updates)}\n")
    print(f"the estimated average out of sample error is {np.average(out_sample_errors)}\n")

#test_hw1p7()

def test_hw2p5and6():
    p = perceptron_2d()
    num_runs = 1000
    num_training_examples = 100
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1

    target = p.rng.uniform(coord_lb, coord_ub, 3)
    training_data = p.random_sample_data(num_training_examples, coord_lb, coord_ub, target)
    X_test, y_test = p.random_sample_data(num_test_examples, coord_lb, coord_ub, target)

    learned = np.empty((num_runs, 3))
    training_errors = np.empty(num_runs)
    test_errors = np.empty(num_runs)

    for i in range(num_runs):
        learned[i], training_errors[i] = p.ls_learning_algo(training_data)
        test_errors[i] = np.count_nonzero(y_test != np.sign(np.dot(X_test, learned[i]))) / num_test_examples

    avg_training_error = np.average(training_errors)
    avg_test_error = np.average(test_errors)


    print(f"the average classification error using least squares learning algorithm in sample is {avg_training_error}\n")
    print(f"the average classification error using least squares learning algorithm out of sample is {avg_test_error}\n")
    

#test_hw2p5and6()

def test_hw2p7():
    p = perceptron_2d()
    num_iter = 100000
    num_runs = 1000
    num_training_examples = 10
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    max_updates = 1000
    update_factor = 1
    
    target = p.rng.uniform(coord_lb, coord_ub, 3)
    training_data = p.random_sample_data(num_training_examples, coord_lb, coord_ub, target)
    X_test, y_test = p.random_sample_data(num_test_examples, coord_lb, coord_ub, target)

    learned = np.empty((num_runs, 3))
    num_updates = np.empty(num_runs)

    init_errors = np.empty(num_runs)
    in_sample_errors = np.empty(num_runs)
    out_sample_errors = np.empty(num_runs)
    
    for run_i in range(num_runs):
        start, init_error = p.ls_learning_algo(training_data)
        init_errors[run_i] = init_error
        learned[run_i], num_updates[run_i], in_sample_errors[run_i] = p.adaline_variant_learning_algo(training_data, num_iter, start, update_factor, max_updates)
        out_sample_errors[run_i] = np.count_nonzero(y_test !=  np.sign(np.dot(X_test, learned[run_i]))) / num_test_examples


    print(f"the average initial error using ls regression is {np.average(init_errors)}\n")
    print(f"the average number of updates before convergence is {np.average(num_updates)}\n")
    print(f"the estimated average out of sample error is {np.average(out_sample_errors)}\n")

#test_hw2p7()


def test_hw2p8():
    p = perceptron_2d()
    num_iter = 100000
    num_runs = 1000
    num_training_examples = 1000
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    max_updates = 1000
    update_factor = 1
    noise_rate = 0.10
    
    def target(x):
        return np.sign((x[:, 1] ** 2) + (x[:, 2] ** 2) - 0.6)
    
    X_train, y_train = p.random_sample_data_nonlinear(num_training_examples, coord_lb, coord_ub, target)

    y_train_noisy = p.make_noise(y_train, noise_rate)

    X_test, y_test = p.random_sample_data_nonlinear(num_test_examples, coord_lb, coord_ub, target)

    learned_ls = np.empty((num_runs, 3))
    num_updates = np.empty(num_runs)

    init_errors = np.empty(num_runs)
    #in_sample_errors = np.empty(num_runs)
    out_sample_errors = np.empty(num_runs)
    
    for run_i in range(num_runs):
        learned_ls[run_i], init_errors[run_i] = p.ls_learning_algo((X_train, y_train_noisy))
        out_sample_errors[run_i] = np.count_nonzero(y_test !=  np.sign(np.dot(X_test, learned_ls[run_i]))) / num_test_examples


    print(f"the average in sample error using ls regression is {np.average(init_errors)}\n")
    #print(f"the average number of updates before convergence is {np.average(num_updates)}\n")
    print(f"the estimated average out of sample error is {np.average(out_sample_errors)}\n")

#test_hw2p8()

def test_hw2p910():
    p = perceptron_2d()
    num_iter = 100000
    num_runs = 1000
    num_training_examples = 1000
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    max_updates = 1000
    update_factor = 1
    noise_rate = 0.10
    
    def target(x):
        return np.sign((x[:, 1] ** 2) + (x[:, 2] ** 2) - 0.6)

    def quad_transformation(x):
        result = np.empty((x.shape[0], 6))
        for i, x_i in enumerate(x):
            result[i] = [1, x_i[1], x_i[2], x_i[1] * x_i[2], x_i[1] ** 2, x_i[2] ** 2]
        
        return result

    
    X_train, y_train = p.random_sample_data_nonlinear(num_training_examples, coord_lb, coord_ub, target)

    X_train_transformed = quad_transformation(X_train)

    y_train_noisy = p.make_noise(y_train, noise_rate)

    X_test, y_test = p.random_sample_data_nonlinear(num_test_examples, coord_lb, coord_ub, target)

    y_test_noisy = p.make_noise(y_test, noise_rate)


    learned_ls = np.empty((num_runs, 6))
    num_updates = np.empty(num_runs)

    init_errors = np.empty(num_runs)
    out_sample_errors = np.empty(num_runs)
    
    for run_i in range(num_runs):
        learned_ls[run_i], init_errors[run_i] = p.ls_learning_algo((X_train_transformed, y_train_noisy))
        out_sample_errors[run_i] = np.count_nonzero(y_test_noisy !=  np.sign(np.dot(quad_transformation(X_test), learned_ls[run_i]))) / num_test_examples


    print(f"the average in sample error using ls regression on the transformed data is is {np.average(init_errors)}\n")
    #print(f"the average number of updates before convergence is {np.average(num_updates)}\n")
    print(f"the estimated average out of sample error on the transformed data is {np.average(out_sample_errors)}\n")
    print(f"the average weight vector hypothesis for ls is {np.average(learned_ls, 0)}\n")

#test_hw2p910()

def test_random_pocket_learning_algo():
    p = perceptron_2d()
    start = np.zeros(3)
    max_updates = 50
    num_iter = (max_updates // 2) * max_updates
    num_runs = 20
    num_training_examples = 100
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    noise_rate = 0.10
    
    target = p.rng.uniform(coord_lb, coord_ub, 3)
    
    X_train, y_train = p.random_sample_data(num_training_examples, coord_lb, coord_ub, target)
    y_train_noisy = p.make_noise(y_train, noise_rate)


    X_test, y_test = p.random_sample_data(num_test_examples, coord_lb, coord_ub, target)
    y_test_noisy = p.make_noise(y_test, noise_rate)


    ein_pocket_sum = np.zeros(max_updates + 1)
    eout_pocket_sum = np.zeros(max_updates + 1)

    ein_sum = np.zeros(max_updates + 1)
    eout_sum = np.zeros(max_updates + 1)
    
    check = True
    for _ in range(num_runs):
        pocket_wts, wts = p.random_pocket_learning_algo((X_train, y_train_noisy), num_iter, start, max_updates, True)
        if check:
            print(f"pocket_wts is: {pocket_wts}\n")
            print(f"wts is: {wts}\n")
            print(f"last pocket training error is {np.mean(np.sign(np.dot(X_train, pocket_wts[-1])) != y_train_noisy)}\n")
            print(f"last pocket test error is {np.mean(np.sign(np.dot(X_test, pocket_wts[-1])) != y_test_noisy)}\n")

            check = False
        
        for i in range(max_updates + 1):
            ein_pocket_sum[i] += np.mean(np.sign(np.dot(X_train, pocket_wts[i])) != y_train_noisy)
            eout_pocket_sum[i] += np.mean(np.sign(np.dot(X_test, pocket_wts[i])) != y_test_noisy)

            ein_sum[i] += np.mean(np.sign(np.dot(X_train, wts[i])) != y_train_noisy)
            eout_sum[i] += np.mean(np.sign(np.dot(X_test, wts[i])) != y_test_noisy)

    
    t = np.arange(max_updates + 1)

    avg_ein_pocket = ein_pocket_sum / num_runs
    avg_eout_pocket = eout_pocket_sum / num_runs

    avg_ein = ein_sum / num_runs
    avg_eout = eout_sum  / num_runs

    _, axs = plt.subplots(1, 2, tight_layout=True, sharey=True)

    axs[0].plot(t, avg_ein, label="ein[w(t)]", color="blue")    
    axs[0].plot(t, avg_ein_pocket, label="ein[pocket_w(t)]", color="gold")
    axs[0].legend()
    

    axs[1].plot(t, avg_eout_pocket, label="eout[pocket_w(t)]", color="red")
    axs[1].plot(t, avg_eout, label="eout[w(t)]", color="green")
    axs[1].legend()

    plt.xlabel("updates")
    plt.ylabel("error")
    plt.show()


#test_random_pocket_learning_algo()



def test_3p1():
    #the different grading metrics for performance (error function) leads to different decision boundaries for classification and regression in how they separate
    #the clamp
    p = perceptron_2d()
    #delta = np.pi / 32
    num_examples = 2000
    rad = 10
    thk = 5
    sep = 5

    #x_offset = rad + (thk / 2)

    X_train, y = p.sample_custom_clamp_target(num_examples, rad, thk, sep)




    # t_pos = np.arange(0, np.pi + delta, delta)

    # t_neg = np.arange(np.pi, 2 * np.pi + delta, delta)


    # inner_pos_x = rad * np.ones_like(t_pos) * np.cos(t_pos)
    # inner_pos_y = rad * np.ones_like(t_pos) * np.sin(t_pos)

    # outer_pos_x = (rad + thk) * np.ones_like(t_pos) * np.cos(t_pos)
    # outer_pos_y = (rad + thk) * np.ones_like(t_pos) * np.sin(t_pos)

    # inner_neg_x = x_offset + rad * np.ones_like(t_neg) * np.cos(t_neg)
    # inner_neg_y = -sep + rad * np.ones_like(t_neg) * np.sin(t_neg)

    # outer_neg_x = x_offset + (rad + thk) * np.ones_like(t_neg) * np.cos(t_neg)
    # outer_neg_y = -sep + (rad + thk) * np.ones_like(t_neg) * np.sin(t_neg)

    # plt.plot(inner_pos_x, inner_pos_y)
    # plt.plot(outer_pos_x, outer_pos_y)

    # plt.plot(inner_neg_x, inner_neg_y)
    # plt.plot(outer_neg_x, outer_neg_y)

    num_iter = 1000
    start = p.rng.random(3)
    w_class, num_updates_class, ein_class = p.random_pocket_learning_algo((np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), y), num_iter, start)
    print(f"in sample error for classification is {ein_class}\n")
    print(f"convergence for classification took {num_updates_class} updates\n")

    w_lin, ein_lin = p.ls_learning_algo((np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), y))
    print(f"in sample error for regression is {ein_lin}\n")

    w_class_x = np.linspace(-2 * (rad + thk), 3 * rad + thk)
    w_class_y = -(w_class[1] / w_class[2]) * w_class_x  - (w_class[0] / w_class[2])

    w_lin_x = np.linspace(-2 * (rad + thk), 3 * rad + thk)
    w_lin_y = -(w_lin[1] / w_lin[2]) * w_lin_x  - (w_lin[0] / w_lin[2])

    plt.plot(w_class_x, w_class_y, label='class boundary')
    plt.plot(w_lin_x, w_lin_y, label='regression boundary')

    plt.scatter(X_train[y == 1][:, 0], X_train[y == 1][:, 1], marker='x', label='+1 class')
    plt.scatter(X_train[y == -1][:, 0], X_train[y == -1][:, 1], marker='o', label='-1 class')

    plt.title(f'classification vs regression using clamp constraint\n')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()

#test_3p1()

def test_3p2():
    #less separable data takes longer for pla to separate. length of time isn't linear either. upto to a certain point the amount of updates needed for convergence
    #drops like an  anvil according to the graph
    p = perceptron_2d()
    num_examples = 2000
    num_iter = 1000
    rad = 10
    thk = 5
    start = p.rng.random(3)

    delta = 0.02
    seps = np.arange(delta, 5 + delta, delta)
    num_iters = np.zeros_like(seps)

    for i, sep in enumerate(seps):
        X_train, y = p.sample_custom_clamp_target(num_examples, rad, thk, sep)
        _, num_updates_class, _ = p.random_pocket_learning_algo((np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), y), num_iter, start)
        num_iters[i] = num_updates_class


    plt.plot(seps, num_iters)

    plt.title(f'number of updates for convergence vs sep clamp constraint')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()

#test_3p2()

def test_3p3():
    #for sep = -5 pla will not converge because data isn't linearly separable
    #the upper bound for number of iterations until convergence is inversely proportional to the square of the "minimum error" of the error function among
    # all the data points I think. the sensitivity of the error surface at the points adjacent to sep has to be very small in order to distinguish 
    # between directions of smaller error and everything else. (for ex. if it isn't small enough
    # then traveling along the linear approx will cause you to "make up and possibly exceed" all the misclassified
    # points you attempt to classify correctly if the step is too big) This is my intuitive understanding of why convergence takes more iterations for small sep
    p = perceptron_2d()
    num_examples = 2000
    max_updates = 100000
    num_iter = (max_updates // 2) * max_updates

    rad = 10
    thk = 5
    sep = -5
    start = p.rng.normal(0, 0.1, 3)

    X_train, y_train = p.sample_custom_clamp_target(num_examples, rad, thk, sep)
    min_wts, wts = p.random_pocket_learning_algo((np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), y_train), num_iter, start, max_updates, True)

    w_lin, ein_lin = p.ls_learning_algo((np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), y_train))

    def third_order_transformation(x):
        result = np.empty((x.shape[0], 10))
        for i, x_i in enumerate(x):
            result[i] = [1, x_i[0], x_i[1], x_i[0] * x_i[1],  x_i[0] ** 2, x_i[1] ** 2, (x_i[0] ** 2) * x_i[1], x_i[0] * (x_i[1] ** 2), x_i[0] ** 3, x_i[1] ** 3]
            
        return result

    X_train_transformed = third_order_transformation(X_train)

    min_wts_transformed, wts_transformed = p.random_pocket_learning_algo((X_train_transformed, y_train), num_iter, start, max_updates, True)
    w_lin_transformed, ein_lin_transformed = p.ls_learning_algo((X_train_transformed, y_train))



    ein_min_wts = np.zeros(max_updates + 1)
    ein_min_wts_transformed = np.zeros(max_updates + 1)

    #ein_wts = np.zeros(max_updates + 1)

    for i, min_wt in enumerate(min_wts):
        ein_min_wts[i] = np.mean(np.sign(np.dot(np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), min_wt)) != y_train)
        ein_min_wts_transformed[i] = np.mean(np.sign(np.dot(X_train_transformed, min_wt)) != y_train)

        #ein_wts[i] = np.mean(np.sign(np.dot(np.transpose(np.array([np.ones(num_examples), X_train[:, 0], X_train[:, 1]])), wt)) != y_train)

    w_class = min_wts[-1]
    
    _, axs = plt.subplots(2, 2, tight_layout=True)


    w_lin_x = np.linspace(-2 * (rad + thk), 3 * rad + thk)
    w_lin_y = -(w_lin[1] / w_lin[2]) * w_lin_x  - (w_lin[0] / w_lin[2])

    w_class_x = np.linspace(-2 * (rad + thk), 3 * rad + thk)
    w_class_y = -(w_class[1] / w_class[2]) * w_class_x  - (w_class[0] / w_class[2])

    axs[0, 1].plot(w_class_x, w_class_y, label='class boundary')
    axs[0, 1].plot(w_lin_x, w_lin_y, label='regression boundary')


    axs[0, 1].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], marker='x', label='+1 class')
    axs[0, 1].scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], marker='o', label='-1 class')

    axs[0, 1].set_title(f'pocket vs regression decision for non separable data\n')
    axs[0, 1].set_xlabel('feature 1')
    axs[0, 1].set_ylabel('feature 2')
    axs[0, 1].grid(True)
    axs[0, 1].legend()



    axs[0, 0].plot(np.arange(0, max_updates + 1), ein_min_wts, label="pocket algorithm")
    axs[0, 0].scatter(1, ein_lin, label="least squares algorithm")

    axs[0, 0].set_title(f'in sample error vs number of updates')
    axs[0, 0].set_xlabel('number of updates')
    axs[0, 0].set_ylabel('in sample error')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    plt.show()

#test_3p3()
    

def test_hw5p89():
    p = perceptron_2d()
    start = np.zeros(3)
    max_epochs = 1000
    descent_rate = 0.01
    num_runs = 100
    num_training_examples = 100
    num_test_examples = 1000
    coord_lb = -1
    coord_ub = 1
    
    eouts = np.zeros(num_runs)
    epoch_counts = np.zeros(num_runs)

    for i in range(num_runs):
        target = p.rng.uniform(coord_lb, coord_ub, 3)
        X_train, y_train = p.random_sample_data(num_training_examples, coord_lb, coord_ub, target)
        X_test, y_test = p.random_sample_data(num_test_examples, coord_lb, coord_ub, target)
        learned, ein, last_epoch = p.logistic_sgd_learning_algo((X_train, y_train), start, descent_rate, max_epochs, 0)
        eouts[i] = p.binary_cross_entropy((X_test, y_test), learned)
        epoch_counts[i] = last_epoch + 1
        #print(f'logistic sgd converged in {epoch_to_stop} epochs with in sample error {ein} and learned boundary {learned}\n')

    print(f'the average test error over {num_runs} runs for logistic sgd is {np.mean(eouts)} taking an average of {np.mean(epoch_counts)} epochs to terminate\n')
#test_hw5p89()
def test():
    wor = np.array([1.5, 1, 1])
    wand = np.array([-1.5, 1, 1])
    x = np.arange(-2, 2, 0.1)
    yor = -(wor[1] / wor[2]) * x  - (wor[0] / wor[2])
    yand = -(wand[1] / wand[2]) * x  - (wand[0] / wand[2])
    plt.plot(x, yor, label="or")
    plt.plot(x, yand, label="and")
    plt.legend()
    plt.show()

test()
