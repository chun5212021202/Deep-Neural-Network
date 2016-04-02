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

		for i in range(depth):	# i layer

			if i == 0 :
				w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, x_dimension)),dtype = theano.config.floatX)))
				b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))
				z.append(T.dot(w[i], x_batch_T) + b[i].dimshuffle(0,'x')) # .dimshuffle(0,'x')
				
			else :
				w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, width)),dtype = theano.config.floatX)))
				b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))
				z.append(T.dot(w[i], a[i-1]) + b[i].dimshuffle(0,'x'))	#	.dimshuffle(0,'x')

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
		#self.parameters = self.w + [self.w_output] + self.b + [self.b_output]
		#self.gradients = self.dw + [self.dw_output] + self.db + [self.db_output]

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
			outputs = [cost],#,T.max(y_hat_batch * -T.log(y.T), axis = 1)
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
		#return T.log(1+T.exp(z))    #Danny_Hit


	def softmax(self, z_output) :
		self.total = T.sum(T.exp(z_output), axis = 0)
		return T.exp(z_output) / self.total

	def MyUpdate_Momentum(self, para, grad, move,sigma) :
		#	print len(self.parameters), len(self.gradients)
		update = [(self.learning_rate, self.learning_rate*self.DECAY)]
		update += [( i, self.MOMENTUM*i - self.learning_rate*j ) for i,j in izip(move, grad)]
		update += [( i, i + self.MOMENTUM*k - self.learning_rate*j ) for i,j,k in izip(para, grad, move)]

		return update

	def MyUpdate_Adagrad(self, para, grad, move,sigma) :	# NOT YET DONE!!!!
		update = []
		
		update += [( i, (i**2 + j**2)**0.5) for i,j in izip(sigma, grad)]
		update += [( i, i - self.learning_rate*j /(k**2+j**2)**0.5) for i,j,k in izip(para, grad, sigma)]

		return update






 ######### Utility #########

	def train(self, training_batch_x, training_batch_y) :	#train one data in a epoch
		
		return self.update_parameter(training_batch_x, training_batch_y)
		#return self.update_parameter(training_batch_x, training_batch_y)[0], np.asarray(self.update_parameter(training_batch_x, training_batch_y)[1])
		#print 'output = ',self.y



	def test(self, testing_data) :

		return self.predict(testing_data)











#	def mapping(self, state) :
#		map_file = open('MLDS_HW1_RELEASE_v1/phones/state_48_39.map','r')
#		state_map = list()
#		state = int(state)
#		for line in map_file.readlines() :
#			line = line.strip()
#			if line.startswith(str(state)) :
#				label = line.split()[1]
#				print label
#				return label

		
#	def calculate_error(self) :
#		delta = []
#		for i in range(self.depth) :
#			if i == 0 :
#				delta.append((T.switch(self.z_output<0,0,1)*self.dw_output).T)	# (width, 1)
#			else :
#				delta.append(T.switch(self.z[-i]<0,0,1)*T.dot(self.w[i].T, delta[i-1]))
#
#			print delta[i]
			


#	def load_model(self) :




#	def save_model(self) :

