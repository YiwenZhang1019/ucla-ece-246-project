import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out = np.hstack((np.ones((N,1)), X))
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_out = np.hstack((np.ones((N,1)), X))
            for i in range(m-1):
                X_power = np.zeros((N,1))
                X_power = X_power + (i + 2)

                new_X = np.power(X,X_power)
                X_out = np.hstack((X_out, new_X))
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            y_pred = self.predict(X)
            y=np.reshape(y,(N,1))
            reg = np.abs(self.w) * self.reg
            loss = (1/N) * np.sum(np.square(y - y_pred)) + np.sum(reg)
            X_out = self.gen_poly_features(X)
            dw_l1_norm = np.sign(self.w) * self.reg
            grad = (np.dot(X_out.T, y_pred - y)) * 2 / N + dw_l1_norm
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            y_pred = self.predict(X)
            y=np.reshape(y,(N,1))
            reg = np.abs(self.w) * self.reg
            loss = (1/N) * np.sum(np.square(y - y_pred)) + np.sum(reg)
            X_out = self.gen_poly_features(X)
            dw_l1_norm = np.sign(self.w) * self.reg
            grad = (np.dot(X_out.T, y_pred - y)) * 2 / N + dw_l1_norm
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            # ================================================================ #
            # YOUR CODE HERE:
            # Sample batch_size elements from the training data for use in gradient descent.  
            # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
            # The indices should be randomly generated to reduce correlations in the dataset.  
            # Use np.random.choice.  It is better to user WITHOUT replacement.
            # ================================================================ #
            num_train = N
            idx = np.random.choice(range(num_train), batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss = 0.0
            grad = np.zeros_like(self.w)
            # ================================================================ #
            # YOUR CODE HERE: 
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.w
            # ================================================================ #
            loss, grad = self.loss_and_grad(X_batch,y_batch)
            self.w = self.w - eta * grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        loss = 0
        N,d = X.shape
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            X_out = self.gen_poly_features(X)
            X_transpose = X_out.T
            y = np.reshape(y,(N,1))
            X_T_y = np.dot(X_transpose, y)
            X_T_X_inv = np.linalg.inv(np.dot(X_transpose, X_out))
            self.w = np.dot(X_T_X_inv, X_T_y)
            y_pred = np.dot(X_out, self.w)
            loss = np.sum(np.square(y_pred - y)) / N


            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_out = self.gen_poly_features(X)
            X_transpose = X_out.T
            y = np.reshape(y,(N,1))
            X_T_y = np.dot(X_transpose, y)
            X_T_X_inv = np.linalg.inv(np.dot(X_transpose, X_out))
            self.w = np.dot(X_T_X_inv, X_T_y)
            y_pred = np.dot(X_out, self.w)
            loss = np.sum(np.square(y_pred - y)) / N
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X_aug=self.gen_poly_features(X)
            # print(self.w)
            y_pred = np.dot(X_aug, self.w)
            # print(y_pred)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_aug=self.gen_poly_features(X)
            # print(self.w)
            y_pred = np.dot(X_aug, self.w)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred