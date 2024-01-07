#!/usr/bin/env python3
import os

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program (should be set to 1 for Parts 1 and 3)
implicit_num_threads = 1
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot
import threading
import time

from tqdm import tqdm
import matplotlib.pyplot as plt

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


# SOME UTILITY FUNCTIONS that you may find to be useful, from my PA3 implementation
# feel free to use your own implementation instead if you prefer
def multinomial_logreg_error(Xs, Ys, W):
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error

def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    WdotX = numpy.dot(W, Xs[:,ii])
    expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0))
    softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis = 0)
    return numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
# END UTILITY FUNCTIONS


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
        # shuffle the training data
        numpy.random.seed(4787)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
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



# SGD + Momentum (adapt from Programming Assignment 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # TODO students should use their implementation from programming assignment 3
    # or adapt this version, which is from my own solution to programming assignment 3
    models = []
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            #if ((ibatch+1) % monitor_period == 0):
              #  models.append(W)
    return W


# SGD + Momentum (No Allocation) => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    W = numpy.copy(W0)
    V = numpy.zeros_like(W0)

    # Pre-allocate arrays for the loop
    WdotX = numpy.zeros((W0.shape[0], B))
    expWdotX = numpy.zeros_like(WdotX)
    softmaxWdotX = numpy.zeros_like(WdotX)
    grad = numpy.zeros_like(W0)
    WdotX_max = numpy.zeros((1, B))
    expWdotX_sum = numpy.zeros((1, B))
    W_gamma = numpy.zeros(W0.shape)
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        X_batches = [numpy.ascontiguousarray(Xs[:, ibatch*B:min((ibatch+1)*B, n)]) for ibatch in range(int(n/B))]
        Y_batches = [numpy.ascontiguousarray(Ys[:, ibatch*B:min((ibatch+1)*B, n)]) for ibatch in range(int(n/B))]
        for ibatch, (X_batch, Y_batch) in enumerate(zip(X_batches, Y_batches)):
            ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            
            numpy.dot(W, X_batch, out=WdotX)
            numpy.amax(WdotX, axis=0, keepdims=True, out=WdotX_max)
            numpy.exp(WdotX - WdotX_max, out=expWdotX)
            numpy.sum(expWdotX, axis=0, keepdims=True, out = expWdotX_sum)
            numpy.divide(expWdotX, expWdotX_sum, out=softmaxWdotX)
            numpy.dot(softmaxWdotX - Y_batch, X_batch.T, out=grad)
            numpy.divide(grad,len(ii), out = grad)
            numpy.multiply(gamma, W, out = W_gamma)
            numpy.add(grad, W_gamma, out = grad)

            numpy.multiply(beta, V, out=V)
            numpy.multiply(alpha, grad, out=grad)
            numpy.subtract(V, grad, out=V)
            numpy.add(W, V, out=W)
           # if ((ibatch+1) % monitor_period == 0):
            #    print(W)
            
            
    return W


# SGD + Momentum (threaded)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO perform any global setup/initialization/allocation (students should implement this)
    W = numpy.copy(W0)
    V = numpy.zeros_like(W0)
    Bt = B // num_threads

    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    grad_accum = [numpy.zeros_like(W0) for _ in range(num_threads)]

    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                iter_barrier.wait()  # Synchronize before starting gradient computation
                start_idx = ibatch * B + ithread * Bt
                end_idx = start_idx + Bt
                ii = range(start_idx, min(end_idx, n))
                if len(ii) > 0:  # Check if there is work for this thread
                    numpy.copyto(grad_accum[ithread], multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))

                iter_barrier.wait()


    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            # Update the model on the main thread
            total_grad = numpy.sum(grad_accum, axis=0)
            V = beta * V - alpha * total_grad / num_threads
            W += V
            # Reset gradient accumulator for the next batch
            for i in range(num_threads):
                grad_accum[i].fill(0)  # Reset gradient accumulator
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W


# SGD + Momentum (No Allocation) in 32-bits => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code
    W = numpy.copy(W0.astype(numpy.float32))
    V = numpy.zeros_like(W, dtype=numpy.float32)

    # Pre-allocate arrays for the loop
    WdotX = numpy.zeros((W.shape[0], B), dtype=numpy.float32)
    expWdotX = numpy.zeros_like(WdotX, dtype=numpy.float32)
    softmaxWdotX = numpy.zeros_like(WdotX, dtype=numpy.float32)
    grad = numpy.zeros_like(W, dtype=numpy.float32)
    WdotX_max = numpy.zeros((1, B), dtype=numpy.float32)
    expWdotX_sum = numpy.zeros((1, B), dtype=numpy.float32)
    W_gamma = numpy.zeros(W0.shape, dtype=numpy.float32)
    print("Running minibatch sequential-scan SGD with momentum (no allocation, float32)")
    for it in tqdm(range(num_epochs)):
        X_batches = [numpy.ascontiguousarray(Xs[:, ibatch*B:min((ibatch+1)*B, n)], dtype=numpy.float32) for ibatch in range(int(n/B))]
        Y_batches = [numpy.ascontiguousarray(Ys[:, ibatch*B:min((ibatch+1)*B, n)], dtype=numpy.float32) for ibatch in range(int(n/B))]
        for ibatch, (X_batch, Y_batch) in enumerate(zip(X_batches, Y_batches)):
            ii = range(ibatch*B, (ibatch+1)*B)
            numpy.dot(W, X_batch, out=WdotX)
            numpy.amax(WdotX, axis=0, keepdims=True, out=WdotX_max)
            numpy.exp(WdotX - WdotX_max, out=expWdotX)
            numpy.sum(expWdotX, axis=0, keepdims=True, out = expWdotX_sum)
            numpy.divide(expWdotX, expWdotX_sum, out=softmaxWdotX)
            numpy.dot(softmaxWdotX - Y_batch, X_batch.T, out=grad)
            numpy.divide(grad,len(ii), out = grad)
            numpy.multiply(gamma, W, out = W_gamma)
            numpy.add(grad, W_gamma, out = grad)

            numpy.multiply(beta, V, out=V)
            numpy.multiply(alpha, grad, out=grad)
            numpy.subtract(V, grad, out=V)
            numpy.add(W, V, out=W)

    return W


# SGD + Momentum (threaded, float32)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code
    W = numpy.copy(W0.astype(numpy.float32))
    V = numpy.zeros_like(W, dtype=numpy.float32)
    Bt = B // num_threads
    gamma = numpy.float32(gamma)
    alpha = numpy.float32(alpha)
    beta = numpy.float32(beta)
    Xs = Xs.astype(numpy.float32)
    Ys = Ys.astype(numpy.float32)

    iter_barrier = threading.Barrier(num_threads + 1)
    grad_accum = [numpy.zeros_like(W, dtype=numpy.float32) for _ in range(num_threads)]

    def thread_main(ithread):
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                iter_barrier.wait()
                start_idx = ibatch * B + ithread * Bt
                end_idx = start_idx + Bt
                ii = range(start_idx, min(end_idx, n))
                if len(ii) > 0:
                    grad = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
                    numpy.copyto(grad_accum[ithread], grad)
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads, float32)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            total_grad = numpy.sum(grad_accum, axis=0, dtype= numpy.float32)
            V = beta * V - alpha * total_grad / num_threads
            W += V
            # Reset gradient accumulator for the next batch
            for i in range(num_threads):
                grad_accum[i].fill(0)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    return W

def plot_algorithm_performance(data, filename="algorithm_performance.png"):
    """
    Plots and saves the performance of different algorithms with an exponential x-axis.

    Parameters:
    data (dict): A dictionary where keys are algorithm names and values are lists of (batch_size, time) tuples.
    filename (str): The filename to save the plot.
    """
    plt.figure(figsize=(10, 6))

    for algorithm, values in data.items():
        batch_sizes, times = zip(*values)
        plt.plot(batch_sizes, times, marker='o', label=algorithm)

    plt.title('Algorithm Performance: Batch Size vs Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (sec)')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xticks([8, 16, 30, 60, 200, 600, 3000], [8, 16, 30, 60, 200, 600, 3000])  # Set specific x-ticks
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    d = 784  # number of features (28x28 pixels)
    c = 10   # number of classes (digits 0-9)
    W0 = numpy.random.randn(c, d) * 0.01
    Bs = [8,16,30,60,200,600,3000]
    """for b in Bs:
        start_time =time.time()
        model1=sgd_mss_with_momentum(Xs_tr, Ys_tr, 0.0001, W0,0.1,0.9, b, 20)
        end_time =time.time()
        t = end_time-start_time
        print(f"total time sgd: {t}")
        start_time =time.time()
        model2=sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, 0.0001,W0 ,0.1,0.9, b, 20 )
        end_time =time.time()
        t = end_time-start_time
        print(f"total time noalloc: {t}")

        start_time =time.time()
        model3=sgd_mss_with_momentum_threaded(Xs_tr, Ys_tr, 0.0001,W0 ,0.1,0.9, b, 20,4)
        end_time =time.time()
        t = end_time-start_time
        print(f"total time noalloc: {t}")

        start_time =time.time()
        model4=sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, 0.0001,W0 ,0.1,0.9, b, 20)
        end_time =time.time()
        t = end_time-start_time
        print(f"total time noalloc: {t}")

        start_time =time.time()
        model5=sgd_mss_with_momentum_threaded_float32(Xs_tr, Ys_tr, 0.0001,W0 ,0.1,0.9, b, 20,4)
        end_time =time.time()
        t = end_time-start_time
        print(f"total time noalloc: {t}")"""


        #accuracy1 = 1 - multinomial_logreg_error(Xs_te, Ys_te, model1)
        #accuracy2 = 1 - multinomial_logreg_error(Xs_te, Ys_te, model2)
        #accuracy3 = 1 - multinomial_logreg_error(Xs_te, Ys_te, model3)
        #accuracy4 = 1 - multinomial_logreg_error(Xs_te, Ys_te, model4)
        #accuracy5 = 1 - multinomial_logreg_error(Xs_te, Ys_te, model5)
        #print(f"Model 1 - Accuracy: {accuracy1}, with batch = {b}")
        #print(f"Model 2 - Accuracy: {accuracy2}, with batch = {b}")
        #print(f"Model 3 - Accuracy: {accuracy3}, with batch = {b}")
        #print(f"Model 4 - Accuracy: {accuracy4}, with batch = {b}")
        #print(f"Model 5 - Accuracy: {accuracy5}, with batch = {b}")

    p1_dict = {"sgd_mss_with_momentum":[(8, 19.7), (16, 12.3), (30, 9.6), (60,8.3), (200,8.2), (600,9.6), (3000,10.4)], 
               "sgd_mss_with_momentum_noalloc":[(8, 17.2), (16, 10.9), (30, 8.2), (60,8.2), (200,8.6), (600,9.5), (3000,9.9)]}
    
    p2_dict = {"sgd_mss_with_momentum":[(8, 19.7), (16, 12.3), (30, 9.6), (60,8.3), (200,8.2), (600,9.6), (3000,10.4)], 
               "sgd_mss_with_momentum_noalloc":[(8, 17.2), (16, 10.9), (30, 8.2), (60,8.2), (200,8.6), (600,9.5), (3000,9.9)],
               "sgd_mss_with_momentum_4cores":[(8, 19.3), (16, 12.3), (30, 10.0), (60,7.3), (200,7.1), (600,8.3), (3000,8.2)], 
               "sgd_mss_with_momentum_noalloc_4cores":[(8, 17.1), (16, 9.8), (30, 8.1), (60,6.8), (200,7.0), (600,7.9), (3000,7.2)]}
    p3_dict = {"sgd_mss_with_momentum":[(8, 19.7), (16, 12.3), (30, 9.6), (60,8.3), (200,8.2), (600,9.6), (3000,10.4)], 
               "sgd_mss_with_momentum_noalloc":[(8, 17.2), (16, 10.9), (30, 8.2), (60,8.2), (200,8.6), (600,9.5), (3000,9.9)],
               "sgd_mss_with_momentum_4cores":[(8, 19.3), (16, 12.3), (30, 10.0), (60,7.3), (200,7.1), (600,8.3), (3000,8.2)], 
               "sgd_mss_with_momentum_noalloc_4cores":[(8, 17.1), (16, 9.8), (30, 8.1), (60,6.8), (200,7.0), (600,7.9), (3000,7.2)],
               "sgd_mss_with_momentum_threaded":[(8, 117.8), (16, 60.7), (30, 34.0), (60,17.5), (200,6.7), (600,4.0), (3000,4.7)]}
    
    p4_dict = {"sgd_mss_with_momentum":[(8, 19.7), (16, 12.3), (30, 9.6), (60,8.3), (200,8.2), (600,9.6), (3000,10.4)], 
               "sgd_mss_with_momentum_noalloc":[(8, 17.2), (16, 10.9), (30, 8.2), (60,8.2), (200,8.6), (600,9.5), (3000,9.9)],
               "sgd_mss_with_momentum_4cores":[(8, 19.3), (16, 12.3), (30, 10.0), (60,7.3), (200,7.1), (600,8.3), (3000,8.2)], 
               "sgd_mss_with_momentum_noalloc_4cores":[(8, 17.1), (16, 9.8), (30, 8.1), (60,6.8), (200,7.0), (600,7.9), (3000,7.2)],
               "sgd_mss_with_momentum_threaded":[(8, 117.8), (16, 60.7), (30, 34.0), (60,17.5), (200,6.7), (600,4.0), (3000,4.7)],
               "sgd_mss_with_momentum_noalloc_float32":[(8, 16.4), (16, 8.2), (30, 7.2), (60,7.1), (200,6.9), (600,6.2), (3000,5.7)],
               "sgd_mss_with_momentum_threaded_float32":[(8, 103.3), (16, 52.7), (30, 28.6), (60,15.0), (200,5.1), (600,2.4), (3000,2.1)]}
    
    p5_dict = {"sgd_mss_with_momentum":[(8, 19.7), (16, 12.3), (30, 9.6), (60,8.3), (200,8.2), (600,9.6), (3000,10.4)], 
               "sgd_mss_with_momentum_noalloc":[(8, 17.2), (16, 10.9), (30, 8.2), (60,8.2), (200,8.6), (600,9.5), (3000,9.9)],
               "sgd_mss_with_momentum_4cores":[(8, 19.3), (16, 12.3), (30, 10.0), (60,7.3), (200,7.1), (600,8.3), (3000,8.2)], 
               "sgd_mss_with_momentum_noalloc_4cores":[(8, 17.1), (16, 9.8), (30, 8.1), (60,6.8), (200,7.0), (600,7.9), (3000,7.2)],
               "sgd_mss_with_momentum_threaded":[(8, 117.8), (16, 60.7), (30, 34.0), (60,17.5), (200,6.7), (600,4.0), (3000,4.7)],
               "sgd_mss_with_momentum_noalloc_float32":[(8, 16.4), (16, 8.2), (30, 7.2), (60,7.1), (200,6.9), (600,6.2), (3000,5.7)],
               "sgd_mss_with_momentum_threaded_float32":[(8, 103.3), (16, 52.7), (30, 28.6), (60,15.0), (200,5.1), (600,2.4), (3000,2.1)],
               "sgd_mss_with_momentum_noalloc_float32_4cores":[(8, 14.5), (16, 8.3), (30, 6.3), (60,4.7), (200,4.1), (600,4.9), (3000,5.1)]}

    plot_algorithm_performance(p5_dict, "p5_plot.png")
    
