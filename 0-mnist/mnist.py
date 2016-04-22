from tensorflow.examples.tutorials.mnist import input_data
#The downloaded dataset is split up into three parts the training set (55000 data points), the test set (10000 data points), and the Cross Validation set (5000 data points)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension

#If you want to assign probabilities to an object being one of several things, use softmax regression
# softmax converts "evidence" into probabilities
# in this example softmax serves as an "activation" or "link" function
# softmax exponentiates its inputs and then normalizes them

#to use tensoflow we need to import it
import tensorflow as tf

#We describe the interacting operations by manipulating symbolic variables
x = tf.placeholder(tf.float32, [None, 784])
#x isn't a specific value, it's a placeholder
#A placeholder is like a parameter in a function
# None means that a dimension can be of any length

#In this case x can be any number of 784 dimensional vectors, these are our MNIST images

#A variable is a modifiable tensor that lives in tensoflow's graph of interacting operations. It can be used and modified by the computation
W = tf.Variable(tf.zeros([784, 10]))
#The weights
b = tf.Variable(tf.zeros([10]))
#The bias
#We pass in the initial value of the variable to tf.Variables

#Now we implement the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#TRAINING
#In order to train our model we need to define what it means for the model to be good or bad
#We use cross-entropy as a cost function

#y_ represents the correct y values for each example
y_ = tf.placeholder(tf.float32, [None, 10])

#Our cross-entropy calculation:
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#y contains the neural networks hypothesized values
#tf.reduce_sum adds up all the elements in the second dimension of y due to the reduction_indices=[1] parameter

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Here we are minimizing "cross_entropy" using gradient descent with a learning rate of 0.5

init = tf.initialize_all_variables()
#initialize all of the variables that we have created

sess = tf.Session()
sess.run(init)
#Launch the model in a session and run the operation that initializes the variables

#Run the training step 1000 times:
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#In each step of the loop we get a batch of 100 random data points
#We run "train_step" and feed in our random data points for the placeholders x and y_
#Using small batches of random data is called stochastic training

#Evaluating our model
#tf.argmax will give you the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#correct_prediction contains a list of booleans of whether or not our neural network's guess was accurate

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#We cast the list of booleans to floats and take the average in order to get our accuracy

#Finally we print the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


