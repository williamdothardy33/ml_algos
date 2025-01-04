import numpy as np
import math
def get_3d_inputs(k, domain, input = ([0] * 3)):
    if k == 3:
        x_1, x_2, x_3 = input
        domain.append([x_1, x_2, x_3])
    else:
        for coord in [0, 1]:
            input[k] = coord
            get_3d_inputs(k + 1, domain, input)

def get_outputs(n, co_domains, output = (['o'] * 8)):
    if n == 8:
        co_domains.append(output.copy())
    else:
        for y_i in ['o', '*']:
            output[n] = y_i
            get_outputs(n + 1, co_domains, output)


def get_funs():
    domain = []
    get_3d_inputs(0, domain)
    x = np.array(domain)

    co_domains = []
    get_outputs(0, co_domains)
    ys = np.array(co_domains)
    
    return (x, ys)

def all_funs_H():
    def factory(x, y):
        mapping = {tuple(x_k): y[k] for k, x_k in enumerate(x)}
        def fun(x):
            return mapping[tuple(x)]
        return fun

    x, ys = get_funs()

    all_funs = [factory(x, y_k) for y_k in ys]

    return all_funs



#x_i needs to be in order for the training data as ys
def possible_targets(training_data):
    x, ys = get_funs()
    _, y = training_data

    mask = np.array([True] * len(ys))

    for i, y_i in enumerate(y):
        for k, y_k in enumerate(ys):
            if y_k[i] is not None:
                if y_k[i] != y_i:
                    mask[k] = False

    
    ys_prime = ys[mask]

    return x, ys_prime

#lower training error than h_a_2, doesn't remember anything
def h_a_1(input):
    return '*'

#higher training error than h_a_1 doesn't remember anything
def h_a_2(input):
    return 'o'

#no training error, captures the entire training set
def h_c(input):
    num_ones = np.count_nonzero(input)
    if num_ones % 2 == 1:
        return '*'
    return 'o'

def h_c_complement(input):
    result = h_c(input)
    if result == '*':
        return 'o'
    return '*'

#picks least error
def learning_algo_a(training_data, H):
    x, y = training_data
    min_error = math.inf
    min_h_idx = -1
    
    for k, h in enumerate(H):
        error = 0
        for i, x_i in enumerate(x):
            if h(x_i) != y[i]:
                error += 1

        if error < min_error:
            min_error = error
            min_h_idx = k

    return min_h_idx


#picks most error
def learning_algo_b(training_data, H):
    x, y = training_data
    max_error = -math.inf
    max_h_idx = -1
    
    for k, h in enumerate(H):
        error = 0
        for i, x_i in enumerate(x):
            if h(x_i) != y[i]:
                error += 1

        if error > max_error:
            max_error = error
            max_h_idx = k

    return max_h_idx

#picks first hypothesis in H
def learning_algo_c(training_data, H):
    min_h_idx = -1
    if len(H) > 0:
        min_h_idx = 0

    return min_h_idx


#picks h that minimizes training error and maximizes disagreement between h and x_or hypothesis
def learning_algo_d(training_data, H):
    current_min = math.inf
    current_max = -math.inf
    min_candidates = []
    max_candidates = []


    x, y = training_data
    
    for k, h in enumerate(H):
        error_min = 0
        for i, x_i in enumerate(x):
            if h(x_i) != y[i]:
                error_min += 1
                
        if error_min == current_min:
            min_candidates.append(k)    

        if error_min < current_min:
            while len(min_candidates) != 0:
                min_candidates.pop()

            current_min = error_min
            min_candidates.append(k)

    print(f"len of min_candidates is {len(min_candidates)}\n")

    for k_min in min_candidates:
        error_max = 0
        for x_j in x:
            if H[k_min](x_j) != h_c(x_j):
                error_max += 1

        if error_max == current_max:
            max_candidates.append(k_min)

        if error_max > current_max:
            while len(max_candidates) != 0:
                max_candidates.pop()

            current_max = error_max
            max_candidates.append(k_min)

    print(f"len of max_candidates is {len(max_candidates)}\n")

    if len(max_candidates) != 0:
        result = max_candidates.pop()
        return result
    return -1


def generalization_a(g, test_inputs, possible_ys):
    agreement_tally = np.zeros(len(possible_ys))

    for k, y_k in enumerate(possible_ys):
        for i, x_i in enumerate(test_inputs):
            if g(x_i) == y_k[i]:
                agreement_tally[k] += 1

    return agreement_tally

def analysis(H, H_label, learning_algo, training_data):
    training_x, training_y = training_data
    target_x, targets_ys = possible_targets((training_x, training_y))
    test_x, test_ys = (target_x[len(training_x):], targets_ys[:, len(training_x):])

    g_idx = learning_algo((training_x, training_y), H)
    tally = generalization_a(H[g_idx], test_x, test_ys)
    score = np.dot(tally, tally)

    print("*************************************************************************************************\n")

    print(f"the in sample input is:\n{training_x}\n")
    print(f"the hypothesis picked by learning algorithm is:\n'{H_label[g_idx]}'\n")
    print(f"the out of sample input is:\n{test_x}\n")
    print(f"the outputs for the out of sample input for the {len(test_ys)} possible targets is:\n{test_ys}\n")
    print(f"the number of points for each possible target that g agrees with are:\n{tally}\n")
    print(f"the score for the hypothesis is {score}\n")

    print("*************************************************************************************************\n")



def test_a():
    training_x, training_y = (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]), np.array(['o', '*', '*', 'o', '*']))
    H = [h_a_1, h_a_2]
    H_label = ["true hypothesis", "false hypothesis"]
    analysis(H, H_label, learning_algo_a, (training_x, training_y))
#test_a()

def test_b():
    training_x, training_y = (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]), np.array(['o', '*', '*', 'o', '*']))
    H = [h_a_1, h_a_2]
    H_label = ["true hypothesis", "false hypothesis"]
    analysis(H, H_label, learning_algo_b, (training_x, training_y))
#test_b()

def test_c():
    training_x, training_y = (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]), np.array(['o', '*', '*', 'o', '*']))
    H = [h_c]
    H_label = ["xor hypothesis"]
    analysis(H, H_label, learning_algo_c, (training_x, training_y))
    result = [H[0](x_i) for x_i in training_x]
    print(f"chosen g on the training data is {result}\n")
#test_c()

def test_d():
    training_x, training_y = (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]), np.array(['o', '*', '*', 'o', '*']))
    H = all_funs_H()
    H_label = [f"{i}" for i in range(len(H))]
    analysis(H, H_label, learning_algo_d, (training_x, training_y))
    result = [H[111](x_i) for x_i in training_x]
    print(f"chosen g on the training data is {result}\n")
#test_d()

def test_generalization():
    training_x, training_y = (np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]), np.array(['o', '*', '*', 'o', '*']))
    target_x, targets_ys = possible_targets((training_x, training_y))
    test_x, test_ys = (target_x[len(training_x):], targets_ys[:, len(training_x):])

    H = [h_a_1, h_a_2, h_c, h_c_complement]
    H_label = ["true hypothesis", "false hypothesis", "xor_hypothesis", "not_xor_hypothesis"]
    for i, h in enumerate(H):
        tally = generalization_a(h, test_x, test_ys)
        score = np.dot(tally, tally)
        print("*************************************************************************************************\n")

        print(f"the in sample input is:\n{training_x}\n")
        print(f"the hypothesis picked by learning algorithm is:\n'{H_label[i]}'\n")
        print(f"the out of sample input is:\n{test_x}\n")
        print(f"the outputs for the out of sample input for the {len(test_ys)} possible targets is:\n{test_ys}\n")
        print(f"the number of points for each possible target that g agrees with are:\n{tally}\n")
        print(f"the score for the hypothesis is {score}\n")

        print("*************************************************************************************************\n")


test_generalization()


    
#there are a couple things I thought about looking at for the above
#1) how does this model compare with others
#2) what is the distribution of correctly classified points given a target function
#3) how complex is the model/picked hypothesis (can it remember anything?)
#4) how does percentage of training examples used to pick the model affect the analysis (in the case the function doesn't remember anything does it matter?)
#if I removed some of the examples labeled '*' the other model would have been picked. I suspect even if the other model was pick the situation is symmetric
#so in that case could I say that the two models are equivalent? (h_a_1, h_a_2)
#how does the distribution of correctly classified points change depending on the chosen g (frequency of number of points given a hypothesis are the same but the actual
#hypothesis for a given frequency differs)
#one thing about the xor hypothesis is that I think it can only tell you about the data it was trained on any target that departs even a little bit from that won't
#agree with the picked hypothesis