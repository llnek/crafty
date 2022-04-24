from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from keras.datasets import mnist
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###part 1
class LinReg:
  def __init__(self, lr=0.01, epochs=800):
    self.lr = lr
    self.epochs = epochs
    self.weights = None
    self.bias = None
  def fit(self, X, y):
    m, n = X.shape
    self.weights = np.zeros((n,1))
    self.bias = 0
    y = y.reshape(m,1)
    losses = []
    for epoch in range(self.epochs):
      y_hat = np.dot(X, self.weights) + self.bias
      #Loss Function
      loss = np.mean((y_hat - y)**2)
      losses.append(loss)
      #The gradient/derivative of weights w.r.t Loss
      dw = (1/m)*np.dot(X.T, (y_hat - y))
      #The gradient/derivative of bias w.r.t Loss
      db = (1/m)*np.sum((y_hat - y))
      #The Update Rule for Gradient descent
      self.weights -= self.lr*dw
      self.bias -= self.lr*db
    return self.weights, self.bias, losses
  def predict(self, X):
    return np.dot(X, self.weights) + self.bias

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def part1():
  # Reading the csv file.
  df = pd.read_csv('data.csv')
  # Displayinng the first five elements of the dataframe.
  df.head(10)
  # Taking the Hours and Scores column of the dataframe as X and y
  # respectively and coverting them to numpy arrays.
  X = np.array(df['Hours']).reshape(-1,1)
  y = np.array(df['Scores'])
  # Plotting the data X(Hours) on x-axis and y(Scores) on y-axis
  #plt.figure(figsize=(8,6)) # figure size
  #plt.scatter(X, y)
  #plt.title('Hours vs Scores')
  #plt.xlabel('X (Input) : Hours')
  #plt.ylabel('y (Target) : Scores')
  #plt.show()
  X_train, X_test, y_train, y_test = X[:20], X[20:], y[:20], y[20:]
  model = LinReg(epochs=100)
  w, b, l = model.fit(X_train,y_train)
  fig = plt.figure(figsize=(8,6))
  plt.scatter(X, y)
  plt.plot(X, model.predict(X))  # X and predictions.
  X_test_preds = model.predict(X_test)
  plt.title('Hours vs Percentage')
  plt.xlabel('X (Input) : Hours')
  plt.ylabel('y (Target) : Scores')
  plt.show()

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def part2():
  def wm(point, X, tau):
    # tau --> bandwidth
    # X --> Training data.
    # point --> the x where we want to make the prediction.
    # m is the No of training examples .
    m = X.shape[0]
    # Initialising W as an identity matrix.
    w = np.mat(np.eye(m))
    # Calculating weights for all training examples [x(i)'s].
    for i in range(m):
      xi = X[i]
      d = (-2 * tau * tau)
      w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d)
    return w

  def predict(X, y, point, tau):
    # m = number of training examples.
    m = X.shape[0]
    # Appending a cloumn of ones in X to add the bias term.
    ## # Just one parameter: theta, that's why adding a column of ones
    #### to X and also adding a 1 for the point where we want to
    #### predict.
    X_ = np.append(X, np.ones(m).reshape(m,1), axis=1)
    # point is the x where we want to make the prediction.
    point_ = np.array([point, 1])
    # Calculating the weight matrix using the wm function we wrote      #  # earlier.
    w = wm(point_, X_, tau)
    # Calculating parameter theta using the formula.
    theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y))
    # Calculating predictions.
    pred = np.dot(point_, theta)
    # Returning the theta and predictions
    return theta, pred

  def plot_predictions(X, y, tau, nval):
    # X --> Training data.
    # y --> Output sequence.
    # nval --> number of values/points for which we are going to predict.
    # tau --> the bandwidth.
    # The values for which we are going to predict.
    # X_test includes nval evenly spaced values in the domain of X.
    X_test = np.linspace(-3, 3, nval)
    # Empty list for storing predictions.
    preds = []
    # Predicting for all nval values and storing them in preds.
    for point in X_test:
      theta, pred = predict(X, y, point, tau)
      preds.append(pred)

    # Reshaping X_test and preds
    X_test = np.array(X_test).reshape(nval,1)
    preds = np.array(preds).reshape(nval,1)

    # Plotting
    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.') # Predictions in red color.
    plt.show()

  np.random.seed(8)
  X = np.random.randn(1000,1)
  y = 2*(X**3) + 10 + 4.6*np.random.randn(1000,1)
  plot_predictions(X, y, 0.08, 100)

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#part 3
def part3():
  def find_theta(X, y):
    m = X.shape[0] # Number of training examples.
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((m,1)), axis=1)
    # reshaping y to (m,1)
    y = y.reshape(m,1)
    # The Normal Equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta

  def predict(X, theta):
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)
    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)
    return preds

  np.random.seed(42)
  X = np.random.randn(500,1)
  y = 2*X + 1 + 1.2*np.random.randn(500,1)
  # Getting the Value of theta using the find_theta function.
  theta = find_theta(X, y)
  # Getting the predictions on X using the predict function.
  preds = predict(X,theta)
  # Plotting the predictions.
  fig = plt.figure(figsize=(8,6))
  plt.plot(X, y, 'b.')
  plt.plot(X, preds, 'c-')
  plt.xlabel('X - Input')
  plt.ylabel('y - target / true')
  plt.show()

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#part4
def part4():
  def loss(y, y_hat):
    # y --> true/target value.
    # y_hat --> hypothesis
    return np.mean((y_hat - y)**2)

  # Calulating gradient of loss w.r.t parameters(weights and bias).
  def gradients(X, y, y_hat):
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis
    # w --> weights (parameter).
    # b --> bias (parameter).
    # m-> number of training examples.
    m = X.shape[0]
    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y))
    return dw, db

  def x_transform(X, degrees):
    # X --> Input.
    # degrees --> A list, We add X^(value) feature to the input
    #             where value is one of the values in the list.
    # making a copy of X.
    t = X.copy()
    # Appending columns of higher degrees to X.
    for i in degrees:
      X = np.append(X, t**i, axis=1)
    return X

  def train(X, y, bs, degrees, epochs, lr):
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # degrees --> A list, We add X^(value) feature to the input
    #             where value is one of the values in the list.
    # lr --> Learning rate.
    # Adding features to input X.
    x = x_transform(X, degrees)
    # m-> number of training examples
    # n-> number of features
    m, n = x.shape
    w = np.zeros((n,1))
    b = 0
    y = y.reshape(m,1)
    losses = []
    # Training loop.
    for epoch in range(epochs):
      for i in range((m-1)//bs + 1):
        # Defining batches.
        start_i = i*bs
        end_i = start_i + bs
        xb = x[start_i:end_i]
        yb = y[start_i:end_i]
        y_hat = np.dot(xb, w) + b
        # Getting the gradients of loss w.r.t parameters.
        dw, db = gradients(xb, yb, y_hat)
        # Updating the parameters.
        w -= lr*dw
        b -= lr*db
        # Calculating loss and appending it in the list.
        l = loss(y, np.dot(x, w) + b)
        losses.append(l)
    # returning weights, bias and losses(List).
    return w, b, losses

  def predict(X, w, b, degrees):
    # X --> Input.
    # w --> weights (parameter).
    # b --> bias (parameter).
    #degrees --> A list, We add X^(value) feature to the input
    #             where value is one of the values in the list.
    # Adding degrees to input X.
    x1 = x_transform(X, degrees)
    return np.dot(x1, w) + b

  def r2_score(y, y_hat):
    return 1 - (np.sum((np.array(y_hat)-np.array(y))**2)/
                np.sum((np.array(y)-np.mean(np.array(y)))**2))

  #####
  np.random.seed(42)
  X = np.random.rand(1000,1)
  y = 5*((X)**(2)) + np.random.rand(1000,1)

  w, b, l = train(X, y, bs=100, degrees=[2], epochs=1000, lr=0.01)
  score= r2_score(y, predict(X, w, b, [2]))
  print(score)
  fig = plt.figure(figsize=(8,6))
  plt.plot(X, y, 'y.')
  plt.plot(X, predict(X, w, b, [2]), 'r.')
  plt.legend(["Data", "Polynomial predictions"])
  plt.xlabel('X - Input')
  plt.ylabel('y - target / true')
  plt.title('Polynomial Regression')
  plt.show()

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def part5():

  def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

  def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

  def gradients(X, y, y_hat):
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).
    # m-> number of training examples.
    m = X.shape[0]
    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y))
    return dw, db

  def plot_decision_boundary(X, w, b):
    # X --> Inputs
    # w --> weights
    # b --> bias
    # The Line is y=mx+c
    # So, Equate mx+c = w.X + b
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -w[0]/w[1]
    c = -b/w[1]
    x2 = m*x1 + c
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([-2, 2])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    plt.plot(x1, x2, 'y-')
    plt.show()

  def normalize(X):
    # X --> Input.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    # Normalizing all the n features of X.
    for i in range(n):
      X = (X - X.mean(axis=0))/X.std(axis=0)
    return X

  def train(X, y, bs, epochs, lr):
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    w = np.zeros((n,1))
    b = 0
    y = y.reshape(m,1)
    # Normalizing the inputs.
    x = normalize(X)
    losses = []
    # Training loop.
    for epoch in range(epochs):
      for i in range((m-1)//bs + 1):
        # Defining batches. SGD.
        start_i = i*bs
        end_i = start_i + bs
        xb = X[start_i:end_i]
        yb = y[start_i:end_i]
        # Calculating hypothesis/prediction.
        y_hat = sigmoid(np.dot(xb, w) + b)
        # Getting the gradients of loss w.r.t parameters.
        dw, db = gradients(xb, yb, y_hat)
        w -= lr*dw
        b -= lr*db
      #####
      l = loss(y, sigmoid(np.dot(X, w) + b))
      losses.append(l)
    # returning weights, bias and losses(List).
    return w, b, losses

  def predict(X):
    # X --> Input.
    # Normalizing the inputs.
    x = normalize(X)
    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)
    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    return np.array(pred_class)

  def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)

  ######
  if True:
    X, y = make_classification(n_features=2, n_redundant=0,
                             n_informative=2, random_state=1,
                             n_clusters_per_class=1)
    # Training
    w, b, l = train(X, y, bs=100, epochs=1000, lr=0.01)
  else:
    X, y = make_moons(n_samples=100, noise=0.24)
    w, b, l = train(X, y, bs=100, epochs=1000, lr=0.01)

  acc=accuracy(y, y_hat=predict(X))
  print(acc)
  plot_decision_boundary(X, w, b)

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def part6():

  def step_func(z):
    return 1.0 if (z > 0) else 0.0

  def perceptron(X, y, lr, epochs):
    # X --> Inputs.
    # y --> labels/target.
    # lr --> learning rate.
    # epochs --> Number of iterations.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    # Initializing parapeters(theta) to zeros. +1 in n+1 for the bias term.
    theta = np.zeros((n+1,1))
    # Empty list to store how many examples were misclassified at every iteration.
    n_miss_list = []
    # Training.
    for epoch in range(epochs):
      n_miss = 0
      # looping for every example.
      for idx, x_i in enumerate(X):
        # Insering 1 for bias, X0 = 1.
        x_i = np.insert(x_i, 0, 1).reshape(-1,1)
        # Calculating prediction/hypothesis.
        y_hat = step_func(np.dot(x_i.T, theta))
        # Updating if the example is misclassified.
        if (np.squeeze(y_hat) - y[idx]) != 0:
            theta += lr*((y[idx] - y_hat)*x_i)
            # Incrementing by 1.
            n_miss += 1
      # Appending number of misclassified examples
      # at every iteration.
      n_miss_list.append(n_miss)
    return theta, n_miss_list

  def plot_decision_boundary(X, theta):
    # X --> Inputs
    # theta --> parameters
    # The Line is y=mx+c
    # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[1]/theta[2]
    c = -theta[0]/theta[2]
    x2 = m*x1 + c
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "r^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Algorithm")
    plt.plot(x1, x2, 'y-')
    plt.show()


  X, y = datasets.make_blobs(n_samples=150,n_features=2,
                           centers=2,cluster_std=1.05,
                           random_state=2)
  if False:
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Random Classification Data with 2 classes')
    plt.show()

  theta, miss_l = perceptron(X, y, 0.5, 100)
  plot_decision_boundary(X, theta)

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def part7():
  def one_hot(y, c):
    # y--> label/ground truth.
    # c--> Number of classes.
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    return y_hot

  def softmax(z):
    # z--> linear part.
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    # Calculating softmax for all examples.
    for i in range(len(z)):
      exp[i] /= np.sum(exp[i])
    return exp

  def fit(X, y, lr, c, epochs):
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    # m-> number of training examples
    # n-> number of features
    m, n = X.shape
    w = np.random.random((n, c))
    b = np.random.random(c)
    losses = []
    # Training loop.
    for epoch in range(epochs):
      # Calculating hypothesis/prediction.
      z = X@w + b
      y_hat = softmax(z)
      # One-hot encoding y.
      y_hot = one_hot(y, c)
      # Calculating the gradient of loss w.r.t w and b.
      w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot))
      b_grad = (1/m)*np.sum(y_hat - y_hot)
      w = w - lr*w_grad
      b = b - lr*b_grad
      # Calculating loss and appending it in the list.
      loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
      losses.append(loss)
      if epoch%100==0:
        print('Epoch {epoch}==> Loss = {loss}'.format(epoch=epoch, loss=loss))
    return w, b, losses

  def predict(X, w, b):
    # X --> Input.
    # w --> weights.
    # b --> bias.
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

  def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)



  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  if False:
    fig = plt.figure(figsize=(10,7))
    for i in range(15):
      ax = fig.add_subplot(3, 5, i+1)
      ax.imshow(train_X[i], cmap=plt.get_cmap('gray'))
      ax.set_title('Label (y): {y}'.format(y=train_y[i]))
      plt.axis('off')
    plt.show()
  ####
  X_train = train_X.reshape(60000,28*28)
  X_test = test_X.reshape(10000,28*28)
  # Normalizing.
  X_train = X_train/255
  X_test = X_test/255
  # Training
  w, b, l = fit(X_train, train_y, lr=0.9, c=10, epochs=1000)
  # Accuracy for training set.
  train_preds = predict(X_train, w, b)
  acc=accuracy(train_y, train_preds)
  print(acc)
  #test
  test_preds = predict(X_test, w, b)
  acc=accuracy(test_y, test_preds)
  print(acc)

  fig = plt.figure(figsize=(15,10))
  for i in range(40):
    ax = fig.add_subplot(5, 8, i+1)
    ax.imshow(test_X[i], cmap=plt.get_cmap('gray'))
    ax.set_title('y: {y}/ y_hat: {y_hat}'.format(y=test_y[i], y_hat=test_preds))
    plt.axis('off')
  plt.show()

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def gd_demo():
  data = np.array([(1,1),(2,3), (4,3),(3,2),(5,5)])
  alpha = 0.01  #learn rate
  # initial estimates
  b = np.zeros(data.shape[1])
  # error = (estimate - y)² == (b0 + b1.dot(x) - y)²
  # minimize over beta0 and beta1: we can remove the ² part, as 2*(derivation) = 0 -> 2 can be dropped
  # d/db0: (b0+b1.dot(x) - y ) * (1) -> 1 as b0 derives (inner derivation) to 1
  # d/db1: (b0+b1.dot(x) - y ) * (x) -> 1 as b1.dot(x) derives (inner derivation) to x
  # note that both derivatives result in a single scalar value as we have a dot product + additions/subtractions
  ones = np.ones((5,1)) # use a 1's-vector as first X column -> adds the intercept
  X = np.hstack((ones, data[:,0].reshape(-1, 1)))
  y = data[:,1]#.reshape(-1,1) # column vector
  for i in range(1000):
    # we could do a nested loop as:
    # "for xi in [0,1]:" too keep X dimensions flexible
    error_b0 = np.dot(np.dot(X,b.T) - y, X[:,0]) # X_i is {1}
    error_b1 = np.dot(np.dot(X,b.T) - y, X[:,1]) # X_i is element of R
    b[0] = (b[0]-alpha*error_b0)
    b[1] = (b[1]-alpha*error_b1)
  # results as in R: b0 = 0.4 and b1 = 0.8
  print(b) # array([0.39996589, 0.80000945])
  ####
  reg = LinearRegression(fit_intercept=True).fit(data[:,0].reshape(-1, 1), data[:,1])
  print(f"{reg.intercept_},{reg.coef_[0]}") # 0.3999999999999999 0.7999999999999999



#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
def main():
  gd_demo()




#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
if __name__ == "__main__":
  main()

