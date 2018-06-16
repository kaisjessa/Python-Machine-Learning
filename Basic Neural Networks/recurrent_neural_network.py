import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

#read the dataset
#one_hot creates output in the form of [0,0,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#number of feed fowards + back propagations
hm_epochs = 10
#number of classes in output
n_classes = 10
#number of samples propagated through the network
batch_size = 128

#28 px per chunk
chunk_size = 28
#28 chunks per 28x28 image
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
	#create dictionaries for hidden layer weights and biases
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}

	#formatting and modifying data
	'''
	e.g., for 5x5 image
	x = np.ones((1, 5, 5)) = np.ones((None, n_chunks, chunk_size))
	x = array([[
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1]
			]])
	After transpose, swap 0th and 1st dimension, so x = np.ones((5, 1, 5)) = np.ones((n_chunks, None, chunk_size))
	x = array([
				[[1, 1, 1, 1, 1]],
				[[1, 1, 1, 1, 1]],
				[[1, 1, 1, 1, 1]],
				[[1, 1, 1, 1, 1]],
				[[1, 1, 1, 1, 1]]
			])
	After reshape, flatten by one dimension
	x = np.ones((5, 5)) = np.ones((n_chunks, chunk_size))

	x = array([
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1],
				[1, 1, 1, 1, 1]
			])

	After split, split into 5 chunks/5 arrays

	x = [
		array([[1, 1, 1, 1, 1]]),
		array([[1, 1, 1, 1, 1]]),
		array([[1, 1, 1, 1, 1]]),
		array([[1, 1, 1, 1, 1]]),
		array([[1, 1, 1, 1, 1]])
	 ]
	'''

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	#create long-short-term-memory cell
	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	print("OUTPUTS:", outputs[-1])

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output

def train_neural_network(x):
	#prediction = output of NN
	prediction = recurrent_neural_network(x)

	#calculate cost ("error" between known value and prediction using cost function)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	#minimize the cost using an optimizer
	optimizer = tf.train.AdamOptimizer().minimize(cost)



	#start the session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0

			for _ in range(int(mnist.train.num_examples / batch_size)):
				#"chunk" through batch size
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				#reshape input data for rnn
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

				#run the functions using input and output data with c as cost
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				#add all the costs in the epoch to find loss for the epoch
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		#comparing the indices of the maximum values in the prediction and correct output
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		#evaluate all the accuracies of the test images with the test labels
		#reshape input for rnn
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)