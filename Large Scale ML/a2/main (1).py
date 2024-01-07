
import os
import numpy
from numpy import random
import scipy
from scipy.special import softmax
import mnist
import pickle
import time

# you can use matplotlib for plotting
import matplotlib
from matplotlib import pyplot as plt

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = numpy.ascontiguousarray(Xs_tr)
        Ys_tr = numpy.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset



# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    z = numpy.dot(W,x)
    numerator = numpy.exp(z) 
    denominator = numpy.sum(numpy.exp(z), axis=0) 
    predictions = numerator/denominator
    log_loss = -numpy.sum(y*numpy.log(predictions))
    regularization = 0.5*gamma*numpy.sum(W**2)
    return log_loss +regularization

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the loss with respect to the model parameters W

    
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    z = numpy.matmul(W, x) #Multiply every row of W by x_i (c * d) (d * n)
    numerator = numpy.exp(z)
    denominator = numpy.sum(numpy.exp(z), axis=0, keepdims=True)#Now we have c x n
    softmax_dist = numerator / denominator
    after_y = softmax_dist - y
    after_y = after_y.reshape((after_y.shape[0],1))
    x = x.reshape((x.shape[0],1))
    softmax_val = numpy.matmul(after_y, x.T)
    regularization = gamma * W
    output = softmax_val + regularization
    return output
    #softmax_val = numpy.matmul(after_y.reshape((after_y.shape[0],1)), x.reshape((1,x.shape[0])))


# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
    

def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    wrong_predictions = 0
    total =Xs.shape[1]
    for i in range(Xs.shape[1]):
        x= Xs[:,i]
        y = numpy.argmax(Ys[:,i])
        z = numpy.dot(W,x)
        numerator = numpy.exp(z) 
        denominator = numpy.sum(numpy.exp(z), axis=0) 
        predictions = numerator/denominator
        y_hat = numpy.argmax(predictions)
        if y!=y_hat:
            wrong_predictions+=1
    return wrong_predictions/total

    

# compute the gradient of the multinomial logistic regression objective on a batch, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the gradient of the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, gamma, W, ii=None):
    """if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    # a starter solution using an average of the example gradients

    (d, n) = Xs.shape
    acc = W * 0.0
    for i in ii:
        acc += multinomial_logreg_grad_i(Xs[:, i], Ys[:, i], gamma, W)
    return acc / len(ii)
    """
    if ii is not None:
        Xs = Xs[:, ii]
        Ys = Ys[:, ii]
    
    z = numpy.matmul(W, Xs) 
    numerator = numpy.exp(z)
    denominator = numpy.sum(numerator, axis=0, keepdims=True) 
    softmax_dist = numerator / denominator 
    
    grad_softmax = softmax_dist - Ys 
    grad_w = numpy.matmul(grad_softmax, Xs.T) 

    regularization = gamma * W

    grad = grad_w + regularization
    
    return grad / Xs.shape[1]

    

# compute the cross-entropy loss of the classifier on a batch, with regularization
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the model cross-entropy loss
def multinomial_logreg_batch_loss(Xs, Ys, gamma, W, ii = None):
    """if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d, n) = Xs.shape
    acc = 0.0
    for i in ii:
        acc += multinomial_logreg_loss_i(Xs[:, i], Ys[:, i], gamma, W)
    return acc / len(ii)"""
    if ii is not None:
        Xs = Xs[:, ii]
        Ys = Ys[:, ii]
    Z = numpy.dot(W, Xs)

    exp_Z = numpy.exp(Z)
    denominators = numpy.sum(exp_Z, axis=0)
    predictions = exp_Z / denominators

    log_losses = -numpy.sum(Ys * numpy.log(predictions), axis=0)

    regularization = 0.5 * gamma * numpy.sum(W**2)

    average_log_loss = numpy.mean(log_losses)
    
    return average_log_loss + regularization






# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq
    # TODO students should implement this
    W_curr = W0.copy()
    output = [W0]
    for i in range(num_iters):
        grad =multinomial_logreg_batch_grad(Xs, Ys,gamma, W_curr)
        W_curr -= alpha*grad
        if (i + 1) % monitor_freq == 0 or i == (num_iters - 1):
            output.append(W_curr)
            W_curr = W_curr.copy()
        
    return output


# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
#                   to do this, you'll want code like the following:
#                     models = []
#                     models.append(W0.copy())   # (you may not need the copy if you don't mutate W0)
#                     ...
#                     for sgd_iteration in ... :
#                       ...
#                       # code to compute a single SGD update step here
#                       ...
#                       if (it % monitor_period == 0):
#                         models.append(W)
    
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    result = [W0]
    W_curr=W0
    T = (num_epochs*n)//B
    for t in range(T):
        batch = list(numpy.random.randint(0,n,B))
        w_list = gradient_descent(Xs[:,batch], Ys[:,batch], gamma, W_curr, alpha, 1, monitor_period)
        W_curr = w_list[-1]
        if (t + 1) % monitor_period == 0:
            result.append(W_curr)
    return result

# ALGORITHM 2: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches

def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    result = [W0]
    W_curr=W0
    counter = 0
    for epoch in range(num_epochs):
        for i in range(n//B):
            batch_range = numpy.arange(B*i,((B*i)+B))
            w_list = gradient_descent(Xs[:,batch_range], Ys[:,batch_range], gamma, W_curr, alpha, 1, monitor_period)
            W_curr = w_list[-1]
            if (counter+ 1) % monitor_period == 0:
                result.append(W_curr)
            counter +=1
    return result


# ALGORITHM 3: run stochastic gradient descent with minibatching and without-replacement sampling
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
def sgd_minibatch_random_reshuffling(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    result = [W0]
    W_curr=W0
    counter = 0
    for epoch in range(num_epochs):
        indices = numpy.arange(0,n)
        numpy.random.shuffle(indices)
        for i in range(n//B):
            batch = indices[B*i:B*i+B]
            w_list = gradient_descent(Xs[:,batch], Ys[:,batch], gamma, W_curr, alpha, 1, monitor_period)
            W_curr = w_list[-1]
            if (counter+ 1) % monitor_period == 0:
                result.append(W_curr)
            counter +=1
    return result

def error_calculation(recorded_Ws, gamma):
    training_errors = []
    test_errors = []
    # training_losses = []
    # test_losses = []

    for W in recorded_Ws:

        training_errors.append(multinomial_logreg_error(Xs_tr, Ys_tr, W))
        test_errors.append(multinomial_logreg_error(Xs_te, Ys_te, W))
        # training_losses.append(multinomial_logreg_batch_loss(Xs_tr, Ys_tr, gamma, W))
        # test_losses.append(multinomial_logreg_batch_loss(Xs_te, Ys_te, gamma, W))
    return training_errors, test_errors, 0, 0

def plot_error(errors, iterations, place):
    iteration_numbers = list(range(0, iterations))

    plt.figure()
    plt.plot(iteration_numbers, errors, label=f"{place} Error")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(f"{place} Error vs. Iterations")
    plt.legend()
    plt.savefig(f"{place}_error.png")
    plt.show()
    print("final loss for graph above= " +  str(errors[-1]))

def plot_loss(loss, iterations, place):
    iteration_numbers = list(range(0, iterations))

    plt.figure()
    plt.plot(iteration_numbers, loss, label=f"{place} Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"{place} Loss vs. Iterations")
    plt.legend()
    plt.savefig(f"{place}_loss.png")
    


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    #print(Xs_tr.shape)
    #print(Ys_tr.shape)
    
    #print(sgd_minibatch(Xs_tr, Ys_tr, 0.0001, numpy.ones((10,784)), 0.05, 60, 1, 100))
    #print(sgd_minibatch_random_reshuffling(Xs_tr, Ys_tr, 0.0001, numpy.ones((10,784)), 0.001, 1, 10, 6000))
    
    # W_s = gradient_descent(Xs_tr, Ys_tr, 0.0001, numpy.ones((10,784)), 1, 1000, 10)
    # number 4 
    """
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    plot_error(training_errors, len(W_s), "Train")
    plot_error(test_errors, len(W_s), "Test")
    plot_loss(training_losses, len(W_s), "Train")
    plot_loss(test_losses, len(W_s), "Test")
    """


    """
    # 5 & 7 ____________________________________________________________________
    t0 = time.time()
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .001, 1, 10, 6000)
    t1 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo1, config 5.2")
    
    plot_error(test_errors, len(W_s), "Test: Algo1, config 5.2")
    
    print("algo 1 time 5.2: = " + str(t1-t0))

    t6 = time.time()
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .05, 60, 10, 100)
    t7 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo1, config 5.3 ")
    
    plot_error(test_errors, len(W_s), "Test: Algo1, config 5.3")
    
    print("algo 1 time: 5.3= " + str(t7-t6))
    
# ______________________________________
    t2 = time.time()
    W_s = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .001, 1, 10, 6000)
    t3 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo2, config 5.2")
    
    plot_error(test_errors, len(W_s), "Test: Algo2, config 5.2")
    
    print("algo 2 time 5.2: = " + str(t3-t2))

    t8 = time.time()
    W_s = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .05, 60, 10, 100)
    t9 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo2, config 5.3")
    
    plot_error(test_errors, len(W_s), "Test: Algo2, config 5.3")
    
    print("algo 2 time: 5.3= " + str(t9-t8))

# __________________________________________

    t4 = time.time()
    W_s = sgd_minibatch_random_reshuffling(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .001, 1, 10, 6000)
    t5 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo3, config 5.2")
    
    plot_error(test_errors, len(W_s), "Test: Algo3, config 5.2")
    
    print("algo 3 time: 5.2 = " + str(t5-t4))

    t10 = time.time()
    W_s = sgd_minibatch_random_reshuffling(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .05, 60, 10, 100)
    t11 = time.time()
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: Algo3, config 5.3")
    
    plot_error(test_errors, len(W_s), "Test: Algo3, config 5.3")
    
    print("algo 3 time: 5.3= " + str(t11-t10))


    #6 ______________________________________________________________________________
    # alpha=.001 , B = 60 ; alpha = .01 ; alpha = .01, B = 25 ; alpha = .1 , B = 60
    """

    print("starting question 6")

    """
    print("batch=25 and alpha= .05")
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .05, 25, 10, 100)
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: a=.05, B=25")
    
    plot_error(test_errors, len(W_s), "Test: a=.05, B=25") 
    

    print("batch=60 and alpha= .1")
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .1, 60, 10, 100)
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: a=.1, B=60")
    
    plot_error(test_errors, len(W_s), "Test: a=.1, B=60") 
    

    print("batch=25 and alpha= .1")
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .1, 25, 10, 100)
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: a=.1, B=25")
    
    plot_error(test_errors, len(W_s), "TestTrain: a=.1, B=25") 
    
    """

    print("batch=60 and alpha= .05")
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .05, 60, 10, 100)
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: a=.05, B=60")
    
    plot_error(test_errors, len(W_s), "Test: a=.05, B=60") 

    print("batch=32 and alpha= .2")
    W_s = sgd_minibatch(Xs_tr, Ys_tr, .0001,numpy.ones((10,784)) , .2 , 32, 10, 100)
    training_errors, test_errors, training_losses, test_losses = error_calculation(W_s, 0.0001)
    
    plot_error(training_errors, len(W_s), "Train: a=.2, B=128")
    
    plot_error(test_errors, len(W_s), "Test:  a=.2 B=128") 



