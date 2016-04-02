import theano
import theano.tensor as T
import numpy as np
from itertools import izip






#########################################################
#														#
#						DNN STRUCTURE					#
#														#
#########################################################

class DNN(object):
	def __init__(self, x_dimension, y_dimension, depth, width, learning_rate, batch_size, momentum, decay):

		self.depth = depth
		self.learning_rate = theano.shared(np.array(learning_rate, dtype = theano.config.floatX))
		self.MOMENTUM = momentum
		self.DECAY = decay
		self.Adagrad_init = True

		y_hat_batch = T.matrix()
		x_batch = T.fmatrix()
		x_batch_T = x_batch.T


 ######### PARAMETERS #########
 		#	hidden layer
		w = []	
		b = []
		a = []
		z = []

		for i in range(depth):	# i-th layer

			if i == 0 :
				w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, x_dimension)),dtype = theano.config.floatX)))
				b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))
				z.append(T.dot(w[i], x_batch_T) + b[i].dimshuffle(0,'x'))
				
			else :
				w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, width)),dtype = theano.config.floatX)))
				b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))
				z.append(T.dot(w[i], a[i-1]) + b[i].dimshuffle(0,'x'))

			a.append(self.activation(z[i]))
		#	output layer
		w_output = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(y_dimension, width)),dtype = theano.config.floatX))	# w for each neuron in last layer (y_dimension neurons per layer, each neurons 'width' w)
		b_output = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(y_dimension)),dtype = theano.config.floatX))	# b for each neuron in last layer ('y_dimension' neurons per layer)
		z_output = T.dot(w_output, a[-1]) + b_output.dimshuffle(0,'x')

		y = self.softmax(z_output)

 ######### CALCULATE ERROR #########

		#	cost function
		cost = T.sum( (y_hat_batch * -T.log(y.T)) ) / batch_size

		#	gradients
		dw = T.grad(cost, w)
		db = T.grad(cost, b)
		dw_output = T.grad(cost, w_output)
		db_output = T.grad(cost, b_output)


 ######### UPDATE #########

		#	update list
		parameters = []
		gradients = []
		for i in range(depth):
			parameters.append(w[i])
			parameters.append(b[i])
			gradients.append(dw[i])
			gradients.append(db[i])
		parameters.append(w_output)
		parameters.append(b_output)
		gradients.append(dw_output)
		gradients.append(db_output)


		#	movement initialize
		movement = []
		sigma = []
		for p in parameters :
			movement.append( theano.shared( np.asarray( np.zeros(p.get_value().shape), dtype =theano.config.floatX )))
			sigma.append( theano.shared( np.asarray( np.zeros(p.get_value().shape), dtype =theano.config.floatX )))

		#	update function
		self.update_parameter = theano.function(
			inputs = [x_batch, y_hat_batch],
			updates = self.MyUpdate_Momentum(parameters, gradients, movement, sigma),
			outputs = [cost],
			allow_input_downcast = True
			)


 ######### PREDICTION #########

		output = T.argmax(y, axis=0)
		self.predict = theano.function(
			inputs = [x_batch],
			outputs = output,
			allow_input_downcast = True
			)
 		

 ######### HELPER #########

	def activation(self, z) :	
		return T.switch(z<0,0.001*z,z)    # ReLU 0.001
		#return T.log(1+T.exp(z))    


	def softmax(self, z_output) :
		self.total = T.sum(T.exp(z_output), axis = 0)
		return T.exp(z_output) / self.total

	def MyUpdate_Momentum(self, para, grad, move,sigma) :
		update = [(self.learning_rate, self.learning_rate*self.DECAY)]
		update += [( i, self.MOMENTUM*i - self.learning_rate*j ) for i,j in izip(move, grad)]
		update += [( i, i + self.MOMENTUM*k - self.learning_rate*j ) for i,j,k in izip(para, grad, move)]

		return update




 ######### Utility #########

	def train(self, training_batch_x, training_batch_y) :	#train one data in a epoch
		
		return self.update_parameter(training_batch_x, training_batch_y)


	def test(self, testing_data) :

		return self.predict(testing_data)
