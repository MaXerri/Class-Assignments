#!/usr/bin/env python3
import os
import math
import matplotlib
import pickle
import numpy 
import scipy.special
import mnist
from tqdm import tqdm
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib import animation
import torch
## you may wish to import other things like torch.nn
import scipy.stats as stats
import time


mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants
### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
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
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,0:50000]
        Ys_tr = Ys_tr[:,0:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + torch.special.erf(u/math.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return torch.exp(-u**2/2.0)/math.sqrt(2.0*math.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in PyTorch)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    # TODO students should implement this
    X_squared = torch.sum(Xs**2, dim=0)
    X_axis = X_squared[:, None]
    Z_squared = torch.sum(Zs**2, dim=0)
    Z_axis = Z_squared[None,:]
    sigma = X_axis + Z_axis - 2*torch.matmul(torch.transpose(Xs, 0, 1), Zs)


    return torch.exp(-gamma*sigma)



# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in PyTorch)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # first, do any work that can be shared among predictions
    # TODO students should implement this
    # next, define a nested function to return
    Sigma = rbf_kernel_matrix(Xs, Xs, gamma)
    Sigma_inv = torch.inverse(Sigma + sigma2_noise * torch.eye(Sigma.size(0)))

    def prediction_mean_and_variance(Xtest):
       k_star = rbf_kernel_matrix(Xs, Xtest.unsqueeze(1), gamma).squeeze()

       # Ensure k_star is a row vector for matrix multiplication
       if k_star.ndim == 0:  # Check if k_star is a scalar
        k_star = k_star.unsqueeze(0).unsqueeze(0)  # Convert scalar to 1x1 matrix
       elif k_star.ndim == 1:
        k_star = k_star.unsqueeze(0)

        # Perform matrix multiplication
       mean = k_star @ Sigma_inv @ Ys

        # Ensure the mean is a scalar tensor
       if mean.numel() == 1:
            mean = mean.squeeze()  # Squeeze to a scalar tensor

        # For RBF kernel, K(x_*, x_*) is 1
       K_xx = torch.tensor(1.0)

        # Calculate the variance
       variance = K_xx + sigma2_noise - k_star @ Sigma_inv @ k_star.T

        # Ensure the variance is a scalar tensor
       if variance.numel() == 1:
            variance = variance.squeeze()  # Squeeze to a scalar tensor

       return mean, variance
    return prediction_mean_and_variance
    
"""
    def prediction_mean_and_variance(Xtest):
        # TODO students should implement this
    # construct mean and variance
    return (mean.reshape(()), variance.reshape(())) # be sure to return scalars!
    #finally, return the nested function
    return prediction_mean_and_variance"""

# compute the probability of improvement (PI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    if torch.is_tensor(stdev):
        stdev = torch.clamp(stdev, min=1e-8)
    elif stdev == 0:
        stdev = 1e-8

    # Standardized difference for PI calculation
    Z = ( Ybest-mean) / stdev

    # Compute the CDF of the standard normal distribution using torch.erf
    PI = 0.5 * (1 + torch.erf(Z / torch.sqrt(torch.tensor(2.0))))
    
    return -PI


# compute the expected improvement (EI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    stdev = torch.tensor(stdev) if not isinstance(stdev, torch.Tensor) else stdev
    stdev = torch.clamp(stdev, min=1e-8)

    # Calculate Z
    Z = (Ybest - mean) / stdev

    # Check for NaN in Z, mean, and stdev, and handle if present
    if torch.isnan(Z) or torch.isnan(mean) or torch.isnan(stdev):
        # Handle NaN case (e.g., return a default value or handle appropriately)
        return torch.tensor([0.0], requires_grad=True)

    # Calculate the PDF and CDF of 'a'
    pdf_Z = torch.distributions.Normal(0, 1).log_prob(Z).exp()
    cdf_Z = torch.distributions.Normal(0, 1).cdf(Z)

    # Compute the Expected Improvement
    ei = -(pdf_Z + Z * cdf_Z) * stdev

    return ei


# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        # TODO students should implement this
        if stdev is None or stdev <= 0:
            stdev = 1e-8
        return mean- (kappa*stdev)
    return A_lcb


# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a torch tensor and returns an expression
# x0            initial value to assign to variable (torch tensor)
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    x = x0.detach().clone()  # create a fresh copy of x0
    x.requires_grad = True   # make it a target for differentiation
    opt = torch.optim.SGD([x], alpha)
    for it in range(num_iters):
        opt.zero_grad()
        f = objective(x)
        f.backward()
        opt.step()
    x.requires_grad = False  # make x no longer require gradients
    return (float(f.item()), x)

# run Bayesian optimization to minimize an objective
#
# objective     objective function; takes a torch tensor, returns a python float scalar
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (a torch tensor, e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    # TODO students should implement this
    Ys = torch.zeros(num_iters)
    Xs = torch.zeros((d,num_iters))

    #total_time = 0
    for i in range(gd_nruns):
        x = random_x()
        #start_time = time.time()
        Ys[i]=objective(x)
        #end_time = time.time()
        #total_time +=(end_time-start_time)
        # Convert x to a PyTorch tensor if it's a NumPy array
        if isinstance(x, numpy.ndarray):
            x_tensor = torch.from_numpy(x).squeeze()
        else:
            # Clone x if it's already a PyTorch tensor
            x_tensor = x.squeeze()

        # Assign the tensor to Xs
        Xs[:, i] = x_tensor


    for i in range(n_warmup, num_iters):
        gp_pred = gp_prediction(Xs[:,:i], Ys[:i], gamma, sigma2_noise)
        def inner_objective(x):
            mean, variance = gp_pred(x)
            variance = torch.tensor(variance) if not torch.is_tensor(variance) else variance
            return acquisition(Ys[:i].min(), mean, torch.sqrt(variance))
        
        best_acq_value = float('inf')
        best_x = None
        for _ in range(gd_nruns):
            x0 = random_x()
            x0_tensor = torch.tensor(x0, dtype=torch.float32) if isinstance(x0, numpy.ndarray) else x0
            acq_value, x_opt = gradient_descent(inner_objective,x0_tensor, gd_alpha, gd_niters)
            if acq_value < best_acq_value:
                best_acq_value = acq_value
                best_x = x_opt

        #start_time = time.time()
        Ys[i]= objective(best_x)
        #end_time = time.time()
        #total_time +=(end_time-start_time)
        Xs[:,i]= best_x.squeeze()

    y_best_index = torch.argmin(Ys)
    y_best = Ys[y_best_index]
    x_best = Xs[:,y_best_index]

    return float(y_best), x_best, Ys, Xs

    



# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1,)
    x = x.item() # convert to a python float
    return (math.cos(8.0*x) - 0.3 + (x-0.5)**2)



# compute the gradient of the multinomial logistic regression objective, with regularization (SIMILAR TO PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 2
    return numpy.dot(scipy.special.softmax(numpy.dot(W, Xs[:,ii]), axis=0) - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 2
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error


# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 3
    (d, n) = Xs.shape
    return -numpy.sum(numpy.log(scipy.special.softmax(numpy.dot(W, Xs), axis=0)) * Ys) / n + (gamma / 2) * (numpy.linalg.norm(W, "fro")**2)


# SGD + Momentum: add momentum to the previous algorithm
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
# returns         the final model, after training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # here is the code from my solution
    # you may also adapt your implementation from PA3 if you prefer
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    niter = 0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            niter += 1
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W)
            W = W + V
    return W


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    (Xs_tr, Ys_tr, Xs_va, Ys_va, _, _) = mnist_dataset
    
    
    def train_and_evaluate(params):
        # Rescale hyperparameters
        gamma = 10 ** (-8 * params[0])
        alpha = 0.5 * params[1]
        beta = params[2]

        # Initialize weights to zero
        W0 = numpy.zeros((10, Xs_tr.shape[0]))  # Assuming 10 classes for MNIST

        # Train the model using SGD with Momentum

        
        Xs_tr_np = Xs_tr.numpy() if isinstance(Xs_tr, torch.Tensor) else Xs_tr
        Ys_tr_np = Ys_tr.numpy() if isinstance(Ys_tr, torch.Tensor) else Ys_tr
        Xs_va_np = Xs_va.numpy() if isinstance(Xs_va, torch.Tensor) else Xs_va
        Ys_va_np = Ys_va.numpy() if isinstance(Ys_va, torch.Tensor) else Ys_va
        g = gamma.numpy() if isinstance(gamma, torch.Tensor) else gamma
        a = alpha.numpy() if isinstance(alpha, torch.Tensor) else alpha
        b = beta.numpy() if isinstance(beta, torch.Tensor) else beta
        W = sgd_mss_with_momentum(Xs_tr_np, Ys_tr_np, g, W0, a, b, B, num_epochs)

        # Check for divergence (non-finite weights)
        if not numpy.isfinite(W).all():
            return 0.1

        # Calculate validation error
        validation_error = multinomial_logreg_error(Xs_va_np, Ys_va_np, W)

        return validation_error - 0.9

    return train_and_evaluate


# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
#
# objective     objective function
# acq           acquisition function
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       torch vector of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file
def animate_predictions(objective, acq, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    acq_eval = []
    acq_Xnext = []
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1)]
        ybest = Ysi.min()
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        pred_acqs = []
        for x_eval in xs_eval:
            XE = x_eval.reshape(1)
            (pred_mean, pred_variance) = gp_pred(XE)
            pred_means.append(float(pred_mean))
            pred_variances.append(float(pred_variance))
            pred_acqs.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))
        mean_eval.append(torch.Tensor(pred_means))
        variance_eval.append(torch.Tensor(pred_variances))
        acq_eval.append(torch.Tensor(pred_acqs))
        if it + 1 != len(Ys):
            XE = Xs[0,it+1].reshape(1)
            (pred_mean, pred_variance) = gp_pred(XE)
            acq_Xnext.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))

    fig = pyplot.figure()
    fig.tight_layout()
    ax = fig.gca()
    ax2 = ax.twinx()

    def animate(i):
        ax.clear()
        ax2.clear()
        ax.set_xlabel("parameter")
        ax.set_ylabel("objective")
        ax2.set_ylabel("acquisiton fxn")
        ax.set_title("Bayes Opt After %d Steps" % (i+1))
        l1 = ax.fill_between(xs_eval, mean_eval[i] + 2.0*torch.sqrt(variance_eval[i]), mean_eval[i] - 2.0*torch.sqrt(variance_eval[i]), color="#eaf1f7")
        l2, = ax.plot(xs_eval, [objective(x.reshape(1)) for x in xs_eval])
        l3, = ax.plot(xs_eval, mean_eval[i], color="r")
        l4 = ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        l5, = ax2.plot(xs_eval, acq_eval[i], color="g", ls=":")
        ax.legend([l2, l3, l5], ["objective", "mean", "acquisition"], loc="upper right")
        if i + 1 == len(Ys):
            return l1, l2, l3, l4, l5
        else:
            l6 = ax2.scatter([Xs[0,i+1]], [acq_Xnext[i]], color="g")
            return l1, l2, l3, l4, l5, l6


    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), interval=600, repeat_delay=1000)

    ani.save(filename)

if __name__ == "__main__":
    # TODO students should implement plotting functions here
    mnist_dataset = load_MNIST_dataset_with_validation_split()
    num_epochs = 5
    B = 500  # Batch size

    # Get the objective function
    objective_function = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)

    # Function to generate random hyperparameters
    def random_x():
        return numpy.random.rand(3)
    d = 3  
    gamma = 10.0  
    sigma2_noise = 0.001 
    kappa = 2.0 
    gd_nruns = 20  
    gd_alpha = 0.01  
    gd_niters = 20  
    n_warmup = 3  
    num_iters = 20
    acquisition_function = lcb_acquisition(kappa)
    #start_time = time.time()
    y_best, x_best, Ys, Xs, obj_time = bayes_opt(
        objective_function,
        d,
        gamma,
        sigma2_noise,
        acquisition_function,
        random_x,
        gd_nruns,
        gd_alpha,
        gd_niters,
        n_warmup,
        num_iters
    )
    #end_time = time.time()
    #ttime = end_time-start_time
    #percentage_t = obj_time/ttime
    #print(f"total time: {ttime}, percent of obj: {percentage_t}, obj_time: {obj_time}")
    
    #def test_random_x():
     #return torch.rand(1)
    #objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    #(y_best, x_best, Ys, Xs) = bayes_opt(test_objective, 1, 10.0, 0.001, lcb_acquisition(.001), test_random_x, 20, 0.01, 20, 3, 20)
    #
    print(y_best)
    print(x_best)
    trained_model = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)(x_best)
    (_, _, _, _, Xs_te, Ys_te) = mnist_dataset
    test_error = multinomial_logreg_error(Xs_te, Ys_te, trained_model)
    print(f"Test Error: {test_error}")
    #print(Ys)
    #print(Xs)
    #
    #Xs_plot = torch.linspace(-0.5, 1.5, steps=256)
    #
    #animate_predictions(test_objective, ei_acquisition, 10.0, 0.001, Ys, Xs, Xs_plot, "bayes_opt_ei.mp4")
