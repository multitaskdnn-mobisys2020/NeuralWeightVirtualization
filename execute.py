from __future__ import print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import importlib
import time
import ctypes
from weight_virtualization import VNN
from weight_virtualization import WeightVirtualization

wv_op = tf.load_op_library('./tf_operation.so')
_weight_loader = ctypes.CDLL('./weight_loader.so')
_weight_loader.get_weight.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int),
	ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0359)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.060)
gpu_options=None

def doInfer(sess, graph, data_set, label=None):
	tensor_x_name = "neuron_0:0"
	x = graph.get_tensor_by_name(tensor_x_name)
	tensor_y_name = "neuron_6:0"
	y = graph.get_tensor_by_name(tensor_y_name)
	keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")

	# infer
	data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
	infer_result = sess.run(y, feed_dict={
		x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})
		#x: data_set_reshaped, keep_prob: 1.0}, options=options, run_metadata=run_metadata)

	if label is not None:
		y_ = graph.get_tensor_by_name("y_:0")
		accuracy = graph.get_tensor_by_name("accuracy:0")
		test_accuracy = sess.run(accuracy, feed_dict={
			x: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
		print("Inference accuracy: %f" % test_accuracy)

def extract_graph_def_to(graph_def, end_node_names):
	for node_name in end_node_names:
		node = find_node_by_name(graph_def, node_name)
		assert node is not None, "No end node named %s" % node_name

	extracted_graph_def = tf.GraphDef()
	removed = []
	for node in graph_def.node:
		for name in end_node_names:
			if node.name == name:
				removed.append(node.name)

	while removed:
		for node in graph_def.node:
			if node.name == removed[0]:
				for input_node in node.input:
					removed.append(input_node)
				extracted_graph_def.node.extend([node])
				graph_def.node.remove(node)
		removed.pop(0)

	del graph_def.node[:]
	graph_def.node.extend(extracted_graph_def.node)

def remove_collection_def(collection_def):
	collection_list = []
	for collection in collection_def:
		collection_list.append(collection)

	for collection in collection_list:
		del collection_def[collection]

def find_node_by_name(graph_def, node_name):
	for node in graph_def.node:
		if node.name == node_name:
			return node
	return None

def get_trainable_variables_names(meta_graph_def):
	trainable_variables_names = []
	with tf.Graph().as_default() as graph:
		tf.train.import_meta_graph(meta_graph_def)
		trainable_variables = tf.trainable_variables()
		for trainable_variable in trainable_variables:
			trainable_variables_names.append(trainable_variable.name)

	assert trainable_variables_names
	return trainable_variables_names

def strip_meta_graph(meta_graph_def, output_node_names):
	with tf.Graph().as_default() as graph:
		tf.train.import_meta_graph(meta_graph_def)
		remove_collection_def(meta_graph_def.collection_def)
	extract_graph_def_to(meta_graph_def.graph_def, output_node_names)

def get_trainable_variables_names(meta_graph_def):
        trainable_variables_names = []
        with tf.Graph().as_default() as graph:
                tf.train.import_meta_graph(meta_graph_def)
                trainable_variables = tf.trainable_variables()
                for trainable_variable in trainable_variables:
                        trainable_variables_names.append(trainable_variable.name)

        assert trainable_variables_names
        return trainable_variables_names

def theirs():
	for i in range(10000):
		vnn_no = np.random.randint(len(vnn_list))
		print('vnn_no:', vnn_no)
		vnn = vnn_list[vnn_no]

		time1 = time.time()
		tf.reset_default_graph()
		with tf.Graph().as_default() as graph:
			saver = tf.train.import_meta_graph(vnn.meta_filepath)
			with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
				saver.restore(sess, vnn.wv_opel_filepath)
				time2 = time.time()
				print('load graph, restore took %0.3f ms' % ((time2-time1)*1000.0))

				time1 = time.time()
				doInfer(sess, tf.get_default_graph(), data_set, label)
				time2 = time.time()
				print('doInfer %0.3f ms' % ((time2-time1)*1000.0))

def init_virtualization(wv, sess):
	vnn_list = []
	for name, vnn in sorted(wv.vnns.items()):
		vnn_list.append(vnn)

	virtual_weight_address = None
	#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	time1 = time.time()
	virtual_weight_address = sess.run(wv_op.init_weight(wv.weight_page))
	time2 = time.time()
	print('virtual_weight address:', virtual_weight_address)
	print('init virtual_weight %0.3f ms' % ((time2-time1)*1000.0))

	page_address_list = []
	vnn_no = 0
	for vnn in vnn_list:
		time1 = time.time()
		page_address = sess.run(wv_op.init_page_table(vnn.weight_page_list))
		time2 = time.time()
		print('[VNN %d] init page table %0.3f ms'
			% (vnn_no, (time2-time1)*1000.0))
		page_address_list.append(page_address)
		vnn_no += 1

	#virtual_weight_address = tf.constant(virtual_weight_address, name='virtual_weight_address')
	page_table_address_list = []
	for i in range(len(page_address_list)):
		page_table_address = tf.constant(page_address_list[i], name='page_table_address/' + str(i))
		page_table_address_list.append(page_table_address)

	for vnn in vnn_list:
		with tf.name_scope(vnn.name):
			time1 = time.time()
			tf.train.import_meta_graph(vnn.meta_filepath)
			time2 = time.time()
			print('import_meta_graph %s %0.3f ms' % (vnn.meta_filepath, (time2-time1)*1000.0))


	weight_address_list = []
	weight_len_list = []

	#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	for i in range(len(vnn_list)):
		train_weights = tf.trainable_variables(scope=vnn_list[i].name)
		weight_address, weight_len = sess.run(wv_op.get_weight_address(train_weights))
		weight_address_list.append(weight_address)
		weight_len_list.append(weight_len)

	time1 = time.time()
	sess.run(tf.global_variables_initializer())
	time2 = time.time()
	print('tf.global_variables_initializer %0.3f ms' % ((time2-time1)*1000.0))

	return vnn_list, weight_address_list, weight_len_list, virtual_weight_address, page_address_list

def execute(graph, sess, vnn, layers, data_set, label=None):
	print("executing", vnn.name)
	keep_prob_input = graph.get_tensor_by_name(vnn.name + "/keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name(vnn.name + "/keep_prob:0")
	x = graph.get_tensor_by_name(vnn.name + "/neuron_0:0")
	y = graph.get_tensor_by_name(vnn.name + "/neuron_" + str(layers-1) + ":0")

	data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
	infer_result = sess.run(y, feed_dict={
		x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})

	if label is not None:
		y_ = graph.get_tensor_by_name(vnn.name + "/y_:0")
		accuracy = graph.get_tensor_by_name(vnn.name + "/accuracy:0")
		test_accuracy = sess.run(accuracy, feed_dict={
			x: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
		print("Inference accuracy: %f" % test_accuracy)

def main():
	wv = WeightVirtualization()

	with tf.Graph().as_default() as graph:
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			vnn_list, weight_address_list, weight_len_list, \
				virtual_weight_address, \
				page_address_list = init_virtualization(wv, sess)

			data_list = [ 'mnist_data', 'GSC_v2_data', 'GTSRB_data',
				'cifar10_data', 'svhn_data' ]
			layer_list = [ 7, 6, 7, 7, 7 ]

			for i in range(10):
				vnn_no = np.random.randint(len(vnn_list))
				print('vnn_no:', vnn_no)

				data = __import__(data_list[vnn_no])
				data_set = data.test_set()[0]#[0:1000]
				label = data.test_set()[1]#[0:1000]

				#with tf.Session() as sess:
				#"""
				time1 = time.time()
				num_of_weight = len(weight_address_list[vnn_no])
				weight_address_list_array_type = ctypes.c_int64 * num_of_weight
				weight_len_list_array_type = ctypes.c_int * num_of_weight
				_weight_loader.get_weight(
					weight_address_list_array_type(*weight_address_list[vnn_no]),
					weight_len_list_array_type(*weight_len_list[vnn_no]),
					ctypes.c_int(num_of_weight),
					ctypes.c_int64(virtual_weight_address),
					ctypes.c_int64(page_address_list[vnn_no]),
					ctypes.c_int(wv.weight_per_page))
				time2 = time.time()
				print('get_weight %0.3f ms' % ((time2-time1)*1000.0))
				#"""

				"""
				train_weights = tf.trainable_variables(scope=vnn_list[i].name)
				saver = tf.train.Saver(train_weights)
				time1 = time.time()
				saver.restore(sess, vnn_list[vnn_no].model_filepath)
				time2 = time.time()
				print('restore %0.3f ms' % ((time2-time1)*1000.0))
				"""

				time1 = time.time()
				execute(tf.get_default_graph(), sess, vnn_list[vnn_no],
					layer_list[vnn_no], data_set, label)
				time2 = time.time()
				print('execute %0.3f ms' % ((time2-time1)*1000.0))

if __name__ == '__main__':
	main()
