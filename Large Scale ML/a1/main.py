
import numpy as np
import time
from collections import deque

### a function to create a unique increasing ID
### note that this is just a quick-and-easy way to create a global order
### it's not the only way to do it
global_order_counter = 0
def get_next_order():
    global global_order_counter
    rv = global_order_counter
    global_order_counter = global_order_counter + 1
    return rv

### a helper function to convert constants into BackproppableArray objects
def to_ba(x):
    if isinstance(x, BackproppableArray):
        return x
    elif isinstance(x, np.ndarray):
        return BackproppableArray(x)
    elif isinstance(x, float):
        return BackproppableArray(np.array(x))
    elif isinstance(x, int):
        return BackproppableArray(np.array(float(x)))
    else:
        raise Exception("could not convert {} to BackproppableArray".format(x))

### a class for an array that can be "packpropped-through"
class BackproppableArray(object):
    # np_array     numpy array that stores the data for this object
    def __init__(self, np_array, dependencies=[]):
        super().__init__()
        self.data = np_array

        # grad holds the gradient, an array of the same shape as data
        # before backprop, grad is None
        # during backprop before grad_fn is called, grad holds the partially accumulated gradient
        # after backprop, grad holds the gradient of the loss (the thing we call backward on)
        #     with respect to this array
        # if you want to use the same array object to call backward twice, you need to re-initialize
        #     grad to zero first
        self.grad = None

        # an counter that increments monotonically over the course of the application
        # we know that arrays with higher order must depend only on arrays with lower order
        # we can use this to order the arrays for backpropagation
        self.order = get_next_order()

        # a list of other BackproppableArray objects on which this array directly depends
        # we'll use this later to decide which BackproppableArray objects need to participate in the backward pass
        self.dependencies = dependencies

    # represents me as a string
    def __repr__(self):
        return "({}, type={})".format(self.data, type(self).__name__)

    # returns a list containing this array and ALL the dependencies of this array, not just
    #    the direct dependencies listed in self.dependencies
    # that is, this list should include this array, the arrays in self.dependencies,
    #     plus all the arrays those arrays depend on, plus all the arrays THOSE arrays depend on, et cetera
    # the returned list must only include each dependency ONCE
    def all_dependencies(self):
        # TODO: (1.1) implement some sort of search to get all the dependencies
        results= {}
        queue = deque()
        queue.append(self)
        while len(queue)>0:
            arr = queue.popleft()
            results[arr] = 1
            for dependency in arr.dependencies:
                if dependency in results:
                    continue
                queue.append(dependency)
        results = list(results.keys())
        return results


    # compute gradients of this array with respect to everything it depends on
    def backward(self):
        # can only take the gradient of a scalar
        assert(self.data.size == 1)

        # depth-first search to find all dependencies of this array
        all_my_dependencies = self.all_dependencies()

        # TODO: (1.2) implement the backward pass to compute the gradients
        #   this should do the following
        #   (1) sort the found dependencies so that the ones computed last go FIRST
        #   (2) initialize and zero out all the gradient accumulators (.grad) for all the dependencies
        #   (3) set the gradient accumulator of this array to 1, as an initial condition
        #           since the gradient of a number with respect to itself is 1
        #   (4) call the grad_fn function for all the dependencies in the sorted reverse order
        order_list = sorted(all_my_dependencies, key= lambda x: x.order, reverse=True)
        for dependency in order_list:
            if isinstance(dependency.data, int):
                dependency.grad = 0
            else:
                dependency.grad = np.zeros(dependency.data.shape)
        if isinstance(self.data, int):
            self.grad = 1
        else:
            self.grad = np.full(self.data.shape, 1)
        for dependency in order_list:
            dependency.grad_fn()
        

    # function that is called to process a single step of backprop for this array
    # when called, it must be the case that self.grad contains the gradient of the loss (the
    #     thing we are differentating) with respect to this array
    # this function should update the .grad field of its dependencies
    #
    # this should just say "pass" for the parent class
    #
    # child classes override this
    def grad_fn(self):
        pass

    # operator overloading
    def __add__(self, other):
        return BA_Add(self, to_ba(other))
    def __sub__(self, other):
        return BA_Sub(self, to_ba(other))
    def __mul__(self, other):
        return BA_Mul(self, to_ba(other))
    def __truediv__(self, other):
        return BA_Div(self, to_ba(other))

    def __radd__(self, other):
        return BA_Add(to_ba(other), self)
    def __rsub__(self, other):
        return BA_Sub(to_ba(other), self)
    def __rmul__(self, other):
        return BA_Mul(to_ba(other), self)
    def __rtruediv__(self, other):
        return BA_Div(to_ba(other), self)

    # TODO (2.2) Add operator overloading for matrix multiplication
    def __matmul__(self, other):
        return BA_MatMul(self, to_ba(other))
    def __rmatmul__(self, other):
        return BA_MatMul(to_ba(other), self)

    def sum(self, axis=None, keepdims=True):
        return BA_Sum(self, axis)

    def reshape(self, shape):
        return BA_Reshape(self, shape)

    def transpose(self, axes = None):
        if axes is None:
            axes = range(self.data.ndim)[::-1]
        return BA_Transpose(self, axes)

# TODO: implement any helper functions you'll need to backprop through vectors
def helper(grad, shape):
        if isinstance(grad, int) and isinstance(shape, int):
            return grad
        if grad.shape != shape:
            sum_axes = []
            if shape == ():
                sum_axes = list(range(len(grad.shape)))
            else:
                reversed_grad_shape = grad.shape[::-1]
                reversed_data_shape = shape[::-1]

                for i, (g_dim, x_dim) in enumerate(zip(reversed_grad_shape, reversed_data_shape)):
                    if g_dim != x_dim:
                        sum_axes.append(len(grad.shape) - 1 - i)
                for i in range(len(grad.shape) - len(shape)):
                    sum_axes.append(i)
            total_axes = tuple(sum_axes)
            grad = np.sum(grad, axis=total_axes, keepdims=True)
            grad = grad.reshape(shape)
        return grad

# a class for an array that's the result of an addition operation
class BA_Add(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.3) improve grad fn for Add
        self.x.grad += helper(self.grad, self.x.data.shape)
        self.y.grad += helper(self.grad, self.y.data.shape)

    

            




# a class for an array that's the result of a subtraction operation
class BA_Sub(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Sub
        self.x.grad += helper(self.grad, self.x.data.shape)
        self.y.grad -= helper(self.grad, self.y.data.shape)
    
   
    
    

# a class for an array that's the result of a multiplication operation
class BA_Mul(BackproppableArray):
    # x * y
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Mul
        self.x.grad += helper(self.grad*self.y.data, self.x.data.shape)
        self.y.grad += helper(self.grad*self.x.data, self.y.data.shape)
    
    

# a class for an array that's the result of a division operation
class BA_Div(BackproppableArray):
    # x / y
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Div
        self.x.grad += helper(self.grad/self.y.data, self.x.data.shape)
        self.y.grad -= helper(self.grad*self.x.data/(self.y.data*self.y.data),self.y.data.shape)

    


# a class for an array that's the result of a matrix multiplication operation
class BA_MatMul(BackproppableArray):
    # x @ y
    def __init__(self, x, y):
        # we only support multiplication of matrices, i.e. arrays with shape of length 2
        assert(len(x.data.shape) == 2)
        assert(len(y.data.shape) == 2)
        super().__init__(x.data @ y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for MatMul
        self.x.grad+=self.grad @ (self.y.data).T
        self.y.grad+=(self.x.data).T @ self.grad


# a class for an array that's the result of an exponential operation
class BA_Exp(BackproppableArray):
    # exp(x)
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Exp
        self.x.grad += self.grad*(np.exp(self.x.data))

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    else:
        return np.exp(x)

# a class for an array that's the result of an logarithm operation
class BA_Log(BackproppableArray):
    # log(x)
    def __init__(self, x):
        super().__init__(np.log(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Log
        self.x.grad +=(self.grad/self.x.data)

def log(x):
    if isinstance(x, BackproppableArray):
        return BA_Log(x)
    else:
        return np.log(x)

# TODO: Add your own function
# END TODO

# a class for an array that's the result of a sum operation
class BA_Sum(BackproppableArray):
    # x.sum(axis, keepdims=True)
    def __init__(self, x, axis):
        super().__init__(x.data.sum(axis, keepdims=True), [x])
        self.x = x
        self.axis = axis

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Sum
        self.x.grad += self.grad

# a class for an array that's the result of a reshape operation
class BA_Reshape(BackproppableArray):
    # x.reshape(shape)
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Reshape
        self.x.grad += self.grad.reshape(self.x.data.shape)

# a class for an array that's the result of a transpose operation
class BA_Transpose(BackproppableArray):
    # x.transpose(axes)
    def __init__(self, x, axes):
        super().__init__(x.data.transpose(axes), [x])
        self.x = x
        self.axes = axes

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Transpose

        self.x.grad += self.grad.transpose(self.axes)


# numerical derivative of scalar function f at x, using tolerance eps
def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps))/(2*eps)

def numerical_grad(f, x, eps=1e-5):
    # TODO: (2.5) implement numerical gradient function
    #       this should compute the gradient by applying something like
    #       numerical_diff independently for each entry of the input x
    if isinstance(x,int):
        return numerical_diff(f, x, eps)
    output_gradient_vector = np.zeros(x.shape)
    for i in range(len(x)):
        top, down = x[i] + eps,x[i]-eps
        output_gradient_vector[i] = (f(top)-f(down))/(2*eps)
    return output_gradient_vector

# automatic derivative of scalar function f at x, using backprop
def backprop_diff(f, x):
    ba_x = to_ba(x)
    fx = f(ba_x)
    fx.backward()
    return ba_x.grad



# class to store test functions
class TestFxs(object):
    # scalar-to-scalar tests
    @staticmethod
    def f1(x):
        return x * 2 + 3

    @staticmethod
    def df1dx(x):
        # TODO (1.4) implement symbolic derivative of f1
        return 2

    @staticmethod
    def f2(x):
        return x * x

    @staticmethod
    def df2dx(x):
        # TODO (1.4) implement symbolic derivative of f2
        return 2*x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        # TODO (1.4) implement symbolic derivative of f3
        u = (x - 2.0)
        return (-(u**2) +1)/ ((u*u + 1.0)**2)

    @staticmethod
    def f4(x):
        return log(exp(x*x / 8 - 3*x + 5) + x)

    # scalar-to-scalar tests that use vectors in the middle
    @staticmethod
    def g1(x):
        a = np.ones(3,dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())

    @staticmethod
    def g2(x):
        a = np.ones((4,5),dtype="float64")
        b = np.arange(20,dtype="float64")
        ax = x - a
        bx = log((x + b)*(x + b)).reshape((4,5)).transpose()
        y = bx @ ax
        return y.sum().reshape(())

    # vector-to-scalar tests
    @staticmethod
    def h1(x):  # takes an input of shape (5,)
        b = np.arange(5,dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())

    # TODO: Add any other test functions you want to use here
    def h2(x):
        b = np.arange(1000,dtype="float64")
        xb = x * b - 19
        return (xb * xb).sum().reshape(())
    # END TODO


if __name__ == "__main__":
    # TODO: Test your code using the provided test functions and your own functions
    x_val = 10

    # 1. Calculate derivative using df1dx
    symbolic_derivative = TestFxs.df1dx(x_val)

    numerical_dervative = numerical_diff(TestFxs.f1, x_val, eps=1e-5)

    # 2. Calculate derivative using backprop_diff
    backprop_derivative = backprop_diff(TestFxs.f1, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {symbolic_derivative}")
    print(f"Numerical Derivative at x={x_val}: {numerical_dervative}")
    print(f"Backprop Derivative at x={x_val}: {backprop_derivative}") 

    symbolic_derivative = TestFxs.df2dx(x_val)
    numerical_dervative = numerical_diff(TestFxs.f2, x_val, eps=1e-5)

    # 2. Calculate derivative using backprop_diff
    backprop_derivative = backprop_diff(TestFxs.f2, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {symbolic_derivative}")
    print(f"Numerical Derivative at x={x_val}: {numerical_dervative}")
    print(f"Backprop Derivative at x={x_val}: {backprop_derivative}")

    symbolic_derivative = TestFxs.df3dx(x_val)
    numerical_dervative = numerical_diff(TestFxs.f3, x_val, eps=1e-5)

    # 2. Calculate derivative using backprop_diff
    backprop_derivative = backprop_diff(TestFxs.f3, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {symbolic_derivative}")
    print(f"Numerical Derivative at x={x_val}: {numerical_dervative}")
    print(f"Backprop Derivative at x={x_val}: {backprop_derivative}")

    numerical_dervative = numerical_diff(TestFxs.f4, x_val, eps=1e-5)

    # 2. Calculate derivative using backprop_diff
    backprop_derivative = backprop_diff(TestFxs.f4, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {numerical_dervative}")
    print(f"Backprop Derivative at x={x_val}: {backprop_derivative}") 


    test_derivate = numerical_grad(TestFxs.g1,x_val)

    # 2. Calculate derivative using backprop_diff
    numerical_derivative = backprop_diff(TestFxs.g1, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {test_derivate}")
    print(f"Backprop Derivative at x={x_val}: {numerical_derivative}")

    # 1. Calculate derivative using df1dx
    test_derivate = numerical_grad(TestFxs.g2,x_val)

    # 2. Calculate derivative using backprop_diff
    numerical_derivative = backprop_diff(TestFxs.g2, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {test_derivate}")
    print(f"Backprop Derivative at x={x_val}: {numerical_derivative}")
    x_val = np.arange(5,dtype="float64")

    test_derivate = numerical_grad(TestFxs.h1,x_val)

    # 2. Calculate derivative using backprop_diff
    numerical_derivative = backprop_diff(TestFxs.h1, x_val)

    # 3. Print the results to compare
    print(f"Symbolic Derivative at x={x_val}: {test_derivate}")
    print(f"Backprop Derivative at x={x_val}: {numerical_derivative}")
    x = np.arange(1000,dtype="float64")
    start_time = time.time() 
    numerical_grad(TestFxs.h2,x)
    end_time = time.time()

    elapsed_time1 = end_time - start_time
    
    # 2. Calculate derivative using backprop_diff
    start_time = time.time() 
    numerical_derivative = backprop_diff(TestFxs.h2, x)
    end_time = time.time()

    elapsed_time2 = end_time - start_time

    # 3. Print the results to compare
    print(f"Time of numerical: {elapsed_time1}")
    print(f"Time of backprop: {elapsed_time2}")






