from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import copy
import pickle 
import sys
import argparse
import importlib
import shutil
import time

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.ERROR)
#np.set_printoptions(threshold=np.nan)
network_generator_filename = "network_generator.obj"

class NetworkGenerator:
	def __init__(self):
		self.network_no = 1
		self.network_list = []

	def constructNetwork(self, layers, name=None):
		if name is None:
			network_name = "Network_" + str(self.network_no)
		else:
			network_name = name

		network = Network(self.network_no, layers, network_name)
		self.network_list.append((self.network_no, network, network_name))
		self.network_no += 1

	def destructNetwork(self, network_no):
		print("destructNetwork %d" % network_no)
		for network in self.network_list:
			if network[0] == network_no:
				self.network_list.remove(network)
				if os.path.exists(network[1].network_dir):
					shutil.rmtree(network[1].network_dir)
				self.network_no -= 1
				return

class Network:
	def __init__(self, network_no, layers_str, network_name=None):
		self.vnn = None
		self.network_no = network_no
		self.network_name = network_name
		self.layers = self.parse_layers(layers_str)
		self.layer_type, self.num_of_neuron_per_layer, self.num_of_weight_per_layer,\
			self.num_of_bias_per_layer = self.calculate_num_of_weight(self.layers)

		self.num_of_neuron = 0
		for layer in self.num_of_neuron_per_layer:
			self.num_of_neuron += np.prod(layer)
			
		self.num_of_weight = sum(self.num_of_weight_per_layer)
		self.num_of_bias = sum(self.num_of_bias_per_layer)

		self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
			'network' + str(self.network_no))
		self.network_file_name = 'network' + str(self.network_no)
		self.network_file_path = os.path.join(self.network_dir, self.network_file_name)

		self.neuron_base_name = "neuron_"
		self.weight_base_name = "weight_"
		self.bias_base_name = "bias_"

		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.buildNetwork(sess)
				if not os.path.exists(self.network_dir):
					os.makedirs(self.network_dir)
				self.saveNetwork(sess)

	def buildNetwork(self, sess):
		layer_type = copy.deepcopy(self.layer_type)
		layer_type = list(filter(lambda type: type != 'max_pool', layer_type))
		layers = self.layers
		parameters = {}
		neurons = {}
		parameters_to_regularize = []

		keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		neurons[0] = tf.placeholder(tf.float32, [None]+layers[0],
			name=self.neuron_base_name+'0')
		print(neurons[0])

		for layer_no in range(1, len(layers)):
			weight_name = self.weight_base_name + str(layer_no-1)
			bias_name = self.bias_base_name + str(layer_no-1)
			neuron_name = self.neuron_base_name + str(layer_no)
	
			if layer_type[layer_no] == "conv":
				conv_parameter = {
					'weights': tf.get_variable(weight_name,
						shape=(layers[layer_no]),
						initializer=tf.contrib.layers.xavier_initializer()),
					'biases' : tf.get_variable(bias_name,
						shape=(layers[layer_no][3]),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				#parameters_to_regularize.append(tf.reshape(conv_parameter['weights'],
					#[tf.size(conv_parameter['weights'])]))
				#parameters_to_regularize.append(tf.reshape(conv_parameter['biases'],
					#[tf.size(conv_parameter['biases'])]))

				parameters[layer_no-1] = conv_parameter
				print('conv_parameter', parameters[layer_no-1])

				rank = sess.run(tf.rank(neurons[layer_no-1]))

				for _ in range(4 - rank):
					neurons[layer_no-1] = tf.expand_dims(neurons[layer_no-1], -1)

				# CNN
				strides = 1
				output = tf.nn.conv2d(neurons[layer_no-1],
					conv_parameter['weights'],
					strides=[1, strides, strides, 1], padding='VALID')
				output_biased = tf.nn.bias_add(output, conv_parameter['biases'])

				# max pooling
				k = 2
				neuron = tf.nn.max_pool(tf.nn.leaky_relu(output_biased),
				#neuron = tf.nn.max_pool(tf.nn.sigmoid(output_biased),
					ksize=[1, k, k, 1],
					strides=[1, k, k, 1], padding='VALID', name=neuron_name)
				neurons[layer_no] = neuron

			elif layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":
				fc_parameter = {
					'weights': tf.get_variable(weight_name,
        	                        	shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
                	                        np.prod(self.num_of_neuron_per_layer[layer_no])),
                        	                initializer=tf.contrib.layers.xavier_initializer()), 
					'biases' : tf.get_variable(bias_name,
						shape=(np.prod(self.num_of_neuron_per_layer[layer_no])),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				parameters_to_regularize.append(tf.reshape(fc_parameter['weights'],
					[tf.size(fc_parameter['weights'])]))
				parameters_to_regularize.append(tf.reshape(fc_parameter['biases'],
					[tf.size(fc_parameter['biases'])]))

				parameters[layer_no-1] = fc_parameter
				print('fc_parameter', parameters[layer_no-1])

				# fully-connected
				flattened = tf.reshape(neurons[layer_no-1],
					[-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])]) 
				neuron_drop = tf.nn.dropout(flattened, rate=1 - keep_prob)

				if layer_type[layer_no] == "hidden":
					neuron = tf.nn.leaky_relu(tf.add(tf.matmul(neuron_drop,
						fc_parameter['weights']), fc_parameter['biases']),
						name=neuron_name)
					#neuron = tf.nn.sigmoid(tf.add(tf.matmul(neuron_drop,
						#fc_parameter['weights']), fc_parameter['biases']),
						#name=neuron_name)

				elif layer_type[layer_no] == "output":
					y_b = tf.add(tf.matmul(neuron_drop, fc_parameter['weights']),
						fc_parameter['biases'])
					neuron = tf.divide(tf.exp(y_b-tf.reduce_max(y_b)),
						tf.reduce_sum(tf.exp(y_b-tf.reduce_max(y_b))),
						name=neuron_name)
					#neuron = tf.nn.softmax(tf.matmul(neuron_drop,
						#fc_parameter['weights']) + fc_parameter['biases'],
						#name=neuron_name)

				neurons[layer_no] = neuron
			print(neuron)

		# input
		x = neurons[0]

		# output
		y = neurons[len(layers)-1]

		# correct labels
		y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

		# define the loss function
		regularization = 0.00001 * tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
			reduction_indices=[1]), name='cross_entropy') + regularization

		# define accuracy
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1),
			name='correct_prediction')
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
			name='accuracy')

		# for training
		learning_rate = 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
			name='optimizer').minimize(cross_entropy)

		init = tf.global_variables_initializer()
		sess.run(init)
	
	def loadNetwork(self, sess):
		#time1 = time.time()
		saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
		saver.restore(sess, self.network_file_path)
		#time2 = time.time()
		#print('loadNetwork took %0.3f ms' % ((time2-time1)*1000.0))

	def saveNetwork(self, sess):
		saver = tf.train.Saver()
		saver.save(sess, self.network_file_path)

	def doTrain(self, sess, graph, train_set, validation_set, batch_size,
		train_iteration, optimizer):
		print("doTrain")

		# get tensors
		tensor_x_name = "neuron_0:0"
		x = graph.get_tensor_by_name("neuron_0:0")
		y_ = graph.get_tensor_by_name("y_:0")
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		accuracy = graph.get_tensor_by_name("accuracy:0")

		input_images_validation = validation_set[0]
		input_images_validation_reshaped = np.reshape(validation_set[0],
			([-1] + x.get_shape().as_list()[1:]))
		labels_validation = validation_set[1]

		highest_accuracy = 0
		time1 = time.time()
		# train
		for i in range(train_iteration):
			input_data, labels = self.next_batch(train_set, batch_size)
			input_data_reshpaed = \
				np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

			if i % (100) == 0 or i == (train_iteration-1):
				train_accuracy = sess.run(accuracy, feed_dict={
					x: input_data_reshpaed,
					y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, training accuracy: %f" % (i, train_accuracy))
			
				# validate
				test_accuracy = sess.run(accuracy, feed_dict={
					x: input_images_validation_reshaped, y_: labels_validation,
					keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, Validation accuracy: %f" % (i, test_accuracy))

				if i == 0:
					highest_accuracy = test_accuracy
				else:
					if test_accuracy > highest_accuracy:
						self.saveNetwork(sess)
						highest_accuracy = test_accuracy
						print('saveNetwork for', highest_accuracy)

			sess.run(optimizer, feed_dict={x: input_data_reshpaed,
				y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})

		time2 = time.time()
		print('%0.3f ms' % ((time2-time1)*1000.0))

	def train(self, train_set, validation_set, batch_size, train_iteration):
		print("train")
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.loadNetwork(sess)
				optimizer = graph.get_operation_by_name("optimizer")
				self.doTrain(sess, graph, train_set, validation_set, batch_size,
					train_iteration, optimizer)
				#self.saveNetwork(sess)

	def doInfer(self, sess, graph, data_set, label=None):
		tensor_x_name = "neuron_0:0"
		x = graph.get_tensor_by_name(tensor_x_name)
		tensor_y_name = "neuron_" + str(len(self.layers)-1) + ":0"
		y = graph.get_tensor_by_name(tensor_y_name)
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")

		# infer
		data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
		infer_result = sess.run(y, feed_dict={
			x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})
		#print(infer_result)
		#print(infer_result.shape)

		if label is not None:
			# validate (this is for test)
			y_ = graph.get_tensor_by_name("y_:0")
			accuracy = graph.get_tensor_by_name("accuracy:0")
			test_accuracy = sess.run(accuracy, feed_dict={
				x: data_set_reshaped, y_: label, keep_prob_input: 1.0,
				keep_prob: 1.0})
			print("Inference accuracy: %f" % test_accuracy)

		return infer_result

	def infer(self, data_set, label=None):
		print("infer")
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				#time1 = time.time()
				self.loadNetwork(sess)
				#time2 = time.time()
				#print('Loading parameters took %0.3f ms' % ((time2-time1)*1000.0))
				return self.doInfer(sess, graph, data_set, label)

	def next_batch(self, data_set, batch_size):
		data = data_set[0]
		label = data_set[1] # one-hot vectors

		data_num = np.random.choice(data.shape[0], size=batch_size, replace=False)
		batch = data[data_num,:]
		label = label[data_num,:] # one-hot vectors

		return batch, label

	def parse_layers(self, layers_str):
		layers_list_str = layers_str.split(',')

		layers_list = []
		for layer_str in layers_list_str:
			layer_dimension_list = []
			layer_dimension_list_str = layer_str.split('*')

			for layer_dimension_str in layer_dimension_list_str:
				layer_dimension_list.append(int(layer_dimension_str))

			layers_list.append(layer_dimension_list)

		return layers_list

	def calculate_num_of_weight(self, layers, pad=0, stride=1):
		layer_type = []
		num_of_weight_per_layer = []
		num_of_bias_per_layer = []
		num_of_neuron_per_layer = []

		for layer in layers:
			if layer is layers[0]:
				type = 'input' # input
				layer_type.append(type)
				num_of_neuron_per_layer.append(layer)

			elif layer is layers[-1]:
				type = 'output' # output, fully-connected
				layer_type.append(type)
				num_of_weight = np.prod(layer)*np.prod(num_of_neuron_per_layer[-1])
				num_of_weight_per_layer.append(num_of_weight)
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

			elif len(layer) == 4:
				type = 'conv' # convolutional
				layer_type.append(type)

				num_of_weight_per_layer.append(np.prod(layer))
				num_of_bias_per_layer.append(layer[3])

				h = (num_of_neuron_per_layer[-1][0] - layer[0] + 2*pad) / stride + 1
				w = (num_of_neuron_per_layer[-1][1] - layer[1] + 2*pad) / stride + 1
				d = layer[3]

				max_pool_f = 2
				max_pool_stride = 2

				h_max_pool = (h - max_pool_f) / max_pool_stride + 1
				w_max_pool = (w - max_pool_f) / max_pool_stride + 1
				d_max_pool = d

				num_of_neuron_per_layer.append([h_max_pool,w_max_pool,d_max_pool])
				layer_type.append('max_pool')

			else:
				type = 'hidden' # fully-connected
				layer_type.append(type)
				num_of_weight = np.prod(layer)*np.prod(num_of_neuron_per_layer[-1])
				num_of_weight_per_layer.append(num_of_weight)
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

		print('layer_type:', layer_type)
		print('num_of_neuron_per_layer:', num_of_neuron_per_layer)
		print('num_of_weight_per_layer:', num_of_weight_per_layer)
		print('num_of_bias_per_layer:', num_of_bias_per_layer)

		return [layer_type, num_of_neuron_per_layer,
			num_of_weight_per_layer, num_of_bias_per_layer]

def main(args):
	ng = None
	if os.path.exists(network_generator_filename):
		ng = pickle.load(open(network_generator_filename, 'rb'))
	else:
		ng = NetworkGenerator()

	data = None
	if args.data is not None and args.data != '':
		data = __import__(args.data)

	if args.mode == 'l':
		print('[l] Task')
		for network in ng.network_list:
			print(network)
			print('\tnetwork_no:', network[1].network_no)
			print('\tnetwork_name:', network[1].network_name)
			print('\tnetwork_file_path:', network[1].network_file_path)
			print('\tlayers(%d):' % len(network[1].layers), network[1].layers)
			print('\tlayer_type(%d):'
				% len(network[1].layer_type), network[1].layer_type)
			print('\tnum_of_neuron_per_layer(%d):'
				% network[1].num_of_neuron, network[1].num_of_neuron_per_layer)
			print('\tnum_of_weight_per_layer(%d):'
				% network[1].num_of_weight, network[1].num_of_weight_per_layer)
			print('\tnum_of_bias_per_layer(%d):'
				% network[1].num_of_bias, network[1].num_of_bias_per_layer)

		return

	elif args.mode == 'c':
		print('[c] constructing a network')

		if args.layers == None or args.layers == '':
			print('[c] No layer. Use --layers')
			return

		print('[c] layers:', args.layers)
		ng.constructNetwork(args.layers, args.name)

	elif args.mode == 'd':
		print('[d] destructing a network')
		if args.network_no == -1:
			print('[d] No network_no. Use --network_no')
			return

		ng.destructNetwork(args.network_no)

	elif args.mode == 't':
		print('[t] train')
		if args.network_no == -1:
			print('[t] No network_no. Use --network_no')
			return

		if data == None:
			print('[t] No data. Use --data')
			return

		print('[t] network_no:', args.network_no)
		print('[t] data:', args.data,
			'train/test.size:', data.train_set()[0].shape, data.test_set()[0].shape)

		batch_size = 100
		train_iteration = 5000

		for network in ng.network_list:
			if network[0] == args.network_no:
				network[1].train(data.train_set(), data.test_set(),
					batch_size, train_iteration)

	elif args.mode == 'i':
		print('[i] inference')
		if args.network_no == -1:
			print('[i] No network_no. Use --network_no')
			return

		if data == None:
			print('[i] No data. Use --data')
			return

		for network in ng.network_list:
			if network[0] == args.network_no:
				#time1 = time.time()
				#network[1].infer(data.test_set()[0][0:1], data.test_set()[1][0:10])
				network[1].infer(data.test_set()[0], data.test_set()[1])
				#time2 = time.time()
				#print 'took %0.3f ms' % ((time2-time1)*1000.0)
				return

		print("no network", args.network_no)
		return

	if args.save != False:
		pickle.dump(ng, open(network_generator_filename, 'wb'))

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-mode', type=str,	help='mode', default='l')
	# l: show the current status of NetworkGenerator (-mode=l)
	# c: construct a network (-mode=c -data -layers)
	# d: destruct a network 
	# t: train a network
	# i: inference of a network

	parser.add_argument('-layers', type=str, help='layers', default=None)
	# CNN layer: 4 dimensions (height, width, depth, num of filters) [ex. 3,3,3,8]
	# Fully-connected layer: any dimension less than 4 [ex. 128 / 128*128 / 128*128*1]

	parser.add_argument('-data', type=str, help='data', default=None)
	parser.add_argument('-network_no', type=int, help='network_no', default=-1)
	parser.add_argument('-name', type=str, help='name', default=None)
	parser.add_argument('-save', type=bool, help='save NetworkGenerator?', default=True)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
