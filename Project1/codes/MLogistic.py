import numpy as np

class MLogistic(object):
    def __init__(self, dim=[10,784], reg_param=0):
        """"
        Inputs:
          - dim: dimensions of the weights [number_classes X number_features]
          - reg : Regularization type [L2,L1,L]
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        dim[1] += 1
        self.dim = dim
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
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        X_aug = self.gen_features(X)
        exponential = np.exp(np.dot(X_aug, self.w.T))
        sub_y = np.dot(X_aug, self.w.T)
        #Create y one-hot matrix
        y_hot = np.zeros((y.size, y.max()+1))
        y_hot[np.arange(y.size),y] = 1 
        
        #Total loss
        loss = 1/N * (np.sum(np.log(exponential.sum(axis = 1))) - np.trace(np.dot(sub_y, y_hot.T)))
        
        #Gradient
        softmax = exponential/(exponential.sum(axis = 1))[:,None]
        grad = 1/N * (np.dot((softmax - y_hot).T, X_aug))
        
        #reg
        grad = grad + self.reg*np.sign(self.w)
        loss = loss + self.reg * np.sum(self.w)
        
         
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
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
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w-=eta * grad
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
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_aug = self.gen_features(X)
        num = np.exp(np.dot(X_aug, self.w.T))
        softmax = num/(num.sum(axis = 1))[:,None]
        y_pred = np.argmax(softmax, axis = 1)
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred