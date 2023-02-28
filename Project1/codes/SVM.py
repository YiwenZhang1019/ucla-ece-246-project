import numpy as np

class SVM(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        X_out = np.hstack((np.ones((N,1)), X))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        N,d = X.shape 
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the SVM regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        X_aug = self.gen_features(X)
        h_wx = np.dot(X_aug, self.w)
        res = 1 - y*h_wx
        loss = (1/N) * np.sum(np.maximum(0,res))

        grad = -(1/N)*X_aug.T@(y*(res > 0))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_svm(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N, d = X.shape
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
            idx = np.random.choice(range(N), batch_size, replace=False)
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
            loss, grad = self.loss_and_grad(X_batch, y_batch)
            self.w -= eta*grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labels for the data in X
        """
        y_pred = np.zeros(X.shape[0])
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_aug = self.gen_features(X)
        h = np.dot(X_aug, self.w)
        y_pred = (h > 0).astype('int')
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred