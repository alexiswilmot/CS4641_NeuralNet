import numpy as np

"""
We are going to use the California housing dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2-layer fully connected neural net. We are going to build the neural network from scratch.
"""


class NeuralNet:
    def __init__(
        self,
        y,
        use_dropout,
        use_momentum,
        lr=0.01,
        batch_size=64,
        momentum=0.5,
        dropout_prob=0.3,
    ):
        """
        This method initializes the class, it is implemented for you.
        Args:
            y (np.ndarray): labels
            use_dropout (bool): flag to enable dropout
            use_momentum (bool): flag to use momentum
            lr (float): learning rate
            batch_size (int): batch size to use for training
            momentum (float): momentum to use for training
            dropout_prob (float): dropout probability
        """
        self.y = y  # ground truth labels

        # OTHER HYPERPARAMTERS
        self.y_hat = np.zeros((self.y.shape[0], 3))  # estimated labels
        self.dimensions = [8, 15, 3]  # dimensions of different layers
        self.alpha = 0.05

        # DROPOUT
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob

        # PARAMETERS
        self.parameters = {}  # dictionary for different layer variables
        self.cache = (
            {}
        )  # cache for holding variables during forward propagation to use them in back prop
        self.loss = []  # list to store loss values
        self.batch_y = []  # list of y batched numpy arrays

        # TRAINING HYPERPARAMETERS
        self.iteration = 0  # iterator to index into data for making a batch
        self.batch_size = batch_size  # batch size

        # NEURAL NETWORK INFORMATION
        self.learning_rate = lr  # learning rate
        self.sample_count = self.y.shape[0]  # number of training samples we have
        self._estimator_type = "regression"
        self.neural_net_type = "Leaky Relu -> Softmax"

        # MOMENTUM
        self.use_momentum = use_momentum
        self.momentum = momentum  # momentum factor
        self.change = {}  # dictionary for previous changes for momentum

    def init_parameters(self, param=None):
        """
        This method initializes the neural network variables, it is already implemented for you.
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.

        Args:
            param (dict): Optional dictionary of parameters to use instead of initializing.
        """
        if param is None:
            np.random.seed(0)
            self.parameters["theta1"] = np.random.randn(
                self.dimensions[0], self.dimensions[1]
            ) / np.sqrt(self.dimensions[0])
            self.parameters["b1"] = np.zeros((self.dimensions[1]))
            self.parameters["theta2"] = np.random.randn(
                self.dimensions[1], self.dimensions[2]
            ) / np.sqrt(self.dimensions[1])
            self.parameters["b2"] = np.zeros((self.dimensions[2]))
        else:
            self.parameters = param
            self.parameters["theta1"] = self.parameters["theta1"].T
            self.parameters["theta2"] = self.parameters["theta2"].T
            self.parameters["b1"] = self.parameters["b1"].T
            self.parameters["b2"] = self.parameters["b2"].T

        for layer in self.parameters:
            self.change[layer] = np.zeros_like(self.parameters[layer])

    def leaky_relu(self, alpha, u):
        """
        Performs element wise leaky ReLU.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Do not modify the values in the input in-place; make a copy instead.

        Args:
            alpha (float): slope of negative piece of leaky ReLU
            u (np.ndarray): input with any shape
        Returns:
            o (np.ndarray): output, same shape as input u
        """
        o = np.copy(u)
        o[u < 0] *= alpha
        return o
        # raise NotImplementedError()

    def derivative_leaky_relu(self, alpha, u):
        """
        Compute element-wise differentiation of Leaky ReLU.
        Args:
            u (np.ndarray): input of any dimension
            alpha (float): the slope coefficent of the negative part.
        Returns:
            derivative_leaky_relu(u) (np.ndarray)
        """
        deriv = np.copy(u)
        deriv[u < 0] = alpha
        deriv[u>=0] = 1
        return deriv
        raise NotImplementedError()

    def softmax(self, u):
        """
        Performs softmax function function element-wise.
        To prevent overflow, begin by subtracting each row in u by its maximum!
        Input:
            u (np.ndarray: (N, 3)): logits
        Output:
            o (np.ndarray: (N, 3)): N probability distributions over D classes
        """
        # subtract max value
        maxxy = np.max(u, axis=1, keepdims=True)
        betterU = u - maxxy
        # exp(xi)
        numerator = np.exp(betterU)
        # sum of
        denom = np.sum(np.exp(betterU), axis=1, keepdims=True)
        return np.divide(numerator, denom)
        raise NotImplementedError()

    @staticmethod
    def _dropout(u, prob):
        """
        Implement the dropout layer. Refer to the description for implementation details.
        Args:
            u (np.ndarray: (N, D)): input to dropout layer
            prob: the probability of dropping an unit
        Returns:
            u_after_dropout (np.ndarray: (N, D)): output of dropout layer
            dropout_mask (np.ndarray: (N, D)): dropout mask indicating which units were dropped

        Hint: scale the units after dropout
              use np.random.choice to sample from Bernoulli(prob) the inactivated nodes for each iteration
        """
        # looks at each imput and decide if it will be ser to 0 with some probability p
        # samples from Bernoulli(p) for each unit, resulting in output where pNK of the units are 0
        masky = np.random.choice([0, 1], size=u.shape, p=[prob, 1-prob])
        droppy = u * masky
        if prob != 0:
            # scale by 1/1-p
            droppy /= (1- prob)
        return droppy, masky
        raise NotImplementedError()

    def cross_entropy_loss(self, y, y_hat):
        """
        Computes cross entropy loss.
        Refer to the description in the notebook and implement the appropriate mathematical equation.
        To avoid log(0) errors, add a small constant 1e-15 to the input to np.log
        Args:
            y (np.ndarray: (N, D)): one-hot ground truth labels
            y_hat (np.ndarray: (N, D)): predictions
        Returns:
            loss (float): average cross entropy loss
        """
        # Equation for Cross Entropy Loss: 
            # CS = -(1/N) sigma(y * log y)

        firstPart = -1 / y.shape[0]
        # sigma part
        addedPart = np.sum(np.multiply(y, np.log(y_hat)))
        return firstPart * addedPart
        raise NotImplementedError()

    def forward(self, x, use_dropout):
        """
        Fill in the missing code lines, please refer to the description for more details.
        Check init_parameters method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        Do not change the lines followed by #keep.

        Args:
            x (np.ndarray: (N, 8)): input to neural network
            use_dropout (bool): true if using dropout in forward
        Returns:
            o2 (np.ndarray: (N, 3)): output of neural network

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on the forward pass.
        HINT 2: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        self.cache["X"] = x
        u1 = ...
        o1 = ...
        self.cache["u1"], self.cache["o1"] = u1, o1

        if use_dropout:
            o1 = ...
            dropout_mask = ...
            self.cache["mask"] = dropout_mask

        u2 = ...
        o2 = ...
        self.cache["u2"], self.cache["o2"] = u2, o2
        return o2
        """
        self.cache["X"] = x
        u1 = np.dot(x, self.parameters["theta1"]) + self.parameters["b1"] # theta[1]u[0] + b[1]
        o1 = self.leaky_relu(self.alpha, u1)  # dropout(leakyrelu(u[1]))
        self.cache["u1"], self.cache["o1"] = u1, o1

        if use_dropout:
            o1, dropout_mask = NeuralNet._dropout(o1, self.dropout_prob)
            # dropout_mask = ...
            self.cache["mask"] = dropout_mask

        u2 = np.dot(o1, self.parameters["theta2"]) + self.parameters["b2"] # theta[2]o[1] + b[2]
        o2 = self.softmax(u2) # o2 = y_hat =  self.softmax(u[2])
        self.cache["u2"], self.cache["o2"] = u2, o2
        return o2
        raise NotImplementedError()

    def update_weights(self, dLoss, use_momentum):
        """
        Update weights of neural network based on learning rate given gradients for each layer.
        Can also use momentum to smoothen descent.

        Args:
            dLoss (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)
            use_momentum (bool): flag to use momentum or not

        Return:
            None

        HINT: both self.change and self.parameters need to be updated for use_momentum=True and only self.parameters needs to be updated when use_momentum=False
              momentum records are kept in self.change
        """

        if use_momentum:
            # theta2, b2, theta1, b1
            for paramName, grad in dLoss.items():
                # z_t + 1 = self.momentum * z+t + learning_rate * grad
                # theta_t+1 = theta_t - self.learning_rate * z_t+1
                self.change[paramName] = self.momentum * self.change.get(paramName, 0) + self.learning_rate * grad
                self.parameters[paramName] += self.change[paramName]
        else:
            for paramName, grad in dLoss.items():
                self.parameters[paramName] -= self.learning_rate * grad
        # theta2
        # self.parameters["theta2"] = self.parameters["theta2"] - self.learning_rate * self.derivative_leaky_relu(self.alpha, self.x)
        # # b2
        # self.parameters["b2"] = self.parameters["b2"] - self.learning_rate * 
        # theta1

        # b1

        return
        raise NotImplementedError()

    def compute_gradients(self, y, yh, use_dropout):
        """
        Compute the gradients for each layer given the predicted outputs and ground truths.
        The dropout mask you stored at forward may be helpful.

        Args:
            y (np.ndarray: (N, 3)): ground truth values
            yh (np.ndarray: (N, 3)): predicted outputs

        Returns:
            gradients (dict): dictionary that maps layer names (strings) to gradients (numpy arrays)

        Note: The shapes of the derivatives in gradients are as follows:
            dLoss_theta2 (np.ndarray: (15, 3)): gradients for theta2
            dLoss_b2 (np.ndarray: (3)): gradients for b2
            dLoss_theta1 (np.ndarray: (8, 15)): gradients for theta1
            dLoss_b1 (np.ndarray: (15,)): gradients for b1

        Note: You will have to use the cache (self.cache) to retrieve the values
        from the forward pass!

        HINT 1: Refer to this guide: https://static.us.edusercontent.com/files/gznuqr6aWHD8dPhiusG2TG53 for more detail on computing gradients.

        HINT 2: Division by N only needs to occur ONCE for any derivative that requires a division
        by N. Make sure you avoid cascading divisions by N where you might accidentally divide your
        derivative by N^2 or greater.

        HINT 3: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        dLoss_u2 = yh - y

        dLoss_theta2 = ...
        dLoss_b2 = ...
        dLoss_o1 = ...

        if use_dropout:
            dLoss_u1 = ...
        else:
            dLoss_u1 = ...

        dLoss_theta1 = ...
        dLoss_b1 = ...

        gradients = {"theta1": dLoss_theta1, "b1": dLoss_b1, "theta2": dLoss_theta2, "b2": dLoss_b2}
        return gradients
        """
        dLoss_u2 = yh - y
        N = y.shape[0]
        dLoss_theta2 = np.dot(self.cache["o1"].T, dLoss_u2) / N # dl/du2 * du2/dtheta2
        dLoss_b2 = np.sum(dLoss_u2, axis=0) / N  # dl/db2 = dl/du2 * du2/db2
        dLoss_o1 = np.dot(dLoss_u2, self.parameters["theta2"].T) # dl/do1 = dl/du2 * theta2

        if use_dropout:
            # get dropout mask from cache
            dropout_mask = self.cache["mask"]
            dLoss_u1 = dLoss_o1 * self.derivative_leaky_relu(self.alpha, self.cache["u1"]) * dropout_mask / (1 - self.dropout_prob)# dl/du1 = dl/do1 * d self.leaky_relu()/du1 * dropout_mask * (1/1-p')
                # 1/(1-p) is the scaling factor and dropout_mask is stored in self.cache
        else:
            # no dropout mask
            dLoss_u1 = dLoss_o1 * self.derivative_leaky_relu(self.alpha, self.cache["u1"]) # dl/do1 * d self.leaky_relu/du1
        dLoss_theta1 = np.dot(self.cache["X"].T, dLoss_u1) / N  # dl/du1 * X
        dLoss_b1 = np.sum(dLoss_u1, axis=0) / N  # dl/du1 * 1
        gradients = {"theta1": dLoss_theta1, "b1": dLoss_b1, "theta2": dLoss_theta2, "b2": dLoss_b2}
        return gradients
        # raise NotImplementedError()

    def backward(self, y, yh, use_dropout, use_momentum):
        """
        Fill in the missing code lines, please refer to the description for more details.
        You will need to use cache variables, some of the implemented methods, and other variables as well.
        Refer to the description above and implement the appropriate mathematical equations.
        Do not change the lines followed by #keep.

        Args:
            y (np.ndarray: (N, 3)): ground truth labels
            yh (np.ndarray: (N, 3)): neural network predictions
            use_dropout (bool): flag to use dropout
            use_momentum (bool): flag to use momentum

        Return:
            dLoss_theta2: gradients for theta2
            dLoss_b2: gradients for b2
            dLoss_theta1: gradients for theta1
            dLoss_b1: gradients for b1

        Hint: make calls to compute_gradients and update_weights
        """
        # get gradients
        grads = self.compute_gradients(y, yh, use_dropout)
        # update weights
        self.update_weights(grads, use_momentum)
        firstRet = grads["theta2"]
        secRet = grads["b2"]
        thirdRet = grads["theta1"]
        fourthRet = grads["b1"]
        return firstRet, secRet, thirdRet, fourthRet
        raise NotImplementedError()

    def gradient_descent(self, x, y, iter=60000, use_momentum=False, local_test=False):
        """
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them.
        2. One iteration here is one round of forward and backward propagation on the complete dataset.
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss
        **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        Args:
            x (np.ndarray: N x D): input
            y (np.ndarray: N x 3): ground truth labels
            iter (int): number of iterations to train for
            use_momentum (bool): flag to use momentum or not
            local_test (bool): flag to indicate if local test is being run or not
        """
        # initialize
        self.init_parameters()
        for e in range(iter):
            yHat = self.forward(x, self.use_dropout)
            # loss
            loss = self.cross_entropy_loss(y, yHat)
            # gradient
            grads = self.compute_gradients(y, yHat, use_dropout=self.use_dropout)
            # update weights with gradient
            self.update_weights(grads, use_momentum)
            # local test thing or print every 1000
            if local_test or e % 1000 == 0:
                self.loss.append(loss)
                print(f"Iteration {e}, Loss {loss}")
        return
        raise NotImplementedError()

    # bonus for undergraduate students
    def batch_gradient_descent(self, x, y, use_momentum, iter=60000, local_test=False):
        """
        This function is an implementation of the batch gradient descent algorithm

        Notes:
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient
        2. One iteration here is one round of forward and backward propagation on one minibatch.
           You will use self.iteration and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.

        3. Append and printout loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations.
           **For LOCAL TEST append and print out loss at every iteration instead of every 1000th multiple.

        4. Append the y batched numpy array to self.batch_y at every 1000 iterations i.e. at 0th, 1000th,
           2000th .... iterations. We will use this to determine if batching is done correctly.
           **For LOCAL TEST append the y batched array at every iteration instead of every 1000th multiple

        5. We expect a noisy plot since learning on a batch adds variance to the
           gradients learnt
        6. Be sure that your batch size remains constant (see notebook for more detail). Please
           batch your data in a wraparound manner. For example, given a dataset of 9 numbers,
           [1, 2, 3, 4, 5, 6, 7, 8, 9], and a batch size of 6, the first iteration batch will
           be [1, 2, 3, 4, 5, 6], the second iteration batch will be [7, 8, 9, 1, 2, 3],
           the third iteration batch will be [4, 5, 6, 7, 8, 9], etc...

        Args:
            x (np.ndarray: N x D): input data
            y (np.ndarray: N x 3): ground truth labels
            use_momentum (bool): flag to use momentum or not
            iter (int): number of BATCHES to iterate through
            local_test (bool): True if calling local test, default False for autograder and Q1.3
                    this variable can be used to switch between autograder and local test requirement for
                    appending/printing out loss and y batch arrays
        """
        raise NotImplementedError()

    def predict(self, x):
        """
        This function predicts new data points
        It is implemented for you

        Args:
            x (np.ndarray: (N, M)): input data
        Returns:
            y (np.ndarray: (N,)): predictions
        """
        yh = self.forward(x, False)  # (N, D) = forward((N, M))
        pred = np.argmax(yh, axis=1)  # (N,) = argmax((N, D), axis=1)
        return pred

