from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import copy
import pickle 
import struct
import sys
import argparse
import importlib
import time
from weight_virtualization import VNN
from weight_virtualization import WeightVirtualization
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.client import session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()

#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesLimit
#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import MaxBytesInUse
#from tensorflow.python.client import timeline
#use = BytesInUse()
#limit = BytesLimit()
#peak = MaxBytesInUse()

#options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.RunMetadata()

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0359)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.080)
#gpu_options = None 
gpu_options = tf.GPUOptions(deferred_deletion_bytes=3)
graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
	opt_level=tf.OptimizerOptions.L0))#, do_function_inlining=False))

mod = tf.load_op_library('./tf_operation.so')

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

def doInfer2(graph, sess, vnn, data_set, label=None):
	
	keep_prob_input = graph.get_tensor_by_name(vnn.name + "/keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name(vnn.name + "/keep_prob:0")
	tensor_n0_name = vnn.name + "/neuron_0:0"
	tensor_n1_name = vnn.name + "/neuron_1:0"
	tensor_n2_name = vnn.name + "/neuron_2:0"
	tensor_n3_name = vnn.name + "/neuron_3:0"
	tensor_n4_name = vnn.name + "/neuron_4:0"
	tensor_n5_name = vnn.name + "/neuron_5:0"
	tensor_n6_name = vnn.name + "/neuron_6:0"
	n0 = graph.get_tensor_by_name(tensor_n0_name)
	n1 = graph.get_tensor_by_name(tensor_n1_name)
	n2 = graph.get_tensor_by_name(tensor_n2_name)
	n3 = graph.get_tensor_by_name(tensor_n3_name)
	n4 = graph.get_tensor_by_name(tensor_n4_name)
	n5 = graph.get_tensor_by_name(tensor_n5_name)
	n6 = graph.get_tensor_by_name(tensor_n6_name)


	h = sess.partial_run_setup([n1, n2, n3, n4, n5, n6], [n0, keep_prob, keep_prob_input])
	n1_np = sess.partial_run(h, n1,	feed_dict={n0: data_set_reshaped, keep_prob: 1.0, keep_prob_input: 1.0})
	n2_np = sess.partial_run(h, n2)
	n3_np = sess.partial_run(h, n3)
	n4_np = sess.partial_run(h, n4)
	n5_np = sess.partial_run(h, n5)
	n6_np = sess.partial_run(h, n6)

	#h2 = sess.partial_run_setup([n6], [n5, keep_prob, keep_prob_input])
	#n6_np = sess.partial_run(h2, n6, feed_dict={n5: n5_np, keep_prob: 1.0, keep_prob_input: 1.0})

	if label is not None:
		y_ = graph.get_tensor_by_name(vnn.name + "/y_:0")
		accuracy = graph.get_tensor_by_name(vnn.name + "/accuracy:0")
		test_accuracy = sess.run(accuracy, feed_dict={
			n0: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
		print("Inference accuracy: %f" % test_accuracy)


def ours():
	with tf.Graph().as_default() as graph:
		tf.train.import_meta_graph(vnn.meta_filepath)
		tensor_weights = tf.trainable_variables()
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			time1 = time.time()
			address = sess.run(mod.init_weight(weights))
			time2 = time.time()
			print('address:', address)
			print('init weight %0.3f ms' % ((time2-time1)*1000.0))

			time1 = time.time()
			page_table_address1 = sess.run(mod.init_page_table(vnn1.weight_page_list))
			time2 = time.time()
			print('page_table_address 1:', page_table_address)
			print('init page table %0.3f ms' % ((time2-time1)*1000.0))

			time1 = time.time()
			page_table_address2 = sess.run(mod.init_page_table(vnn2.weight_page_list))
			time2 = time.time()
			print('page_table_address 1:', page_table_address)
			print('init page table %0.3f ms' % ((time2-time1)*1000.0))


		for i in range(3):
			with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
				time1 = time.time()
				sess.run(mod.get_weight(tensor_weights, address, page_table_address,
						page_size=page_size))
				time2 = time.time()
				print('get_weight %0.3f ms' % ((time2-time1)*1000.0))

				time1 = time.time()
				doInfer(sess, tf.get_default_graph(), data_set, label)
				time2 = time.time()
				print('doInfer %0.3f ms' % ((time2-time1)*1000.0))

def theirs(vnn_list):
	for i in range(10000):
		vnn_no = np.random.randint(len(vnn_list))
		print('vnn_no:', vnn_no)
		vnn = vnn_list[vnn_no]

		time1 = time.time()
		tf.reset_default_graph()
		with tf.Graph().as_default() as graph:
			saver = tf.train.import_meta_graph(vnn.meta_filepath)
			with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
				saver.restore(sess, vnn.model_filepath)
				time2 = time.time()
				print('load graph, restore took %0.3f ms' % ((time2-time1)*1000.0))

				time1 = time.time()
				doInfer(sess, tf.get_default_graph(), data_set, label)
				time2 = time.time()
				print('doInfer %0.3f ms' % ((time2-time1)*1000.0))


def init_virtualization2(wv):
	vnn_list = []
	for name, vnn in sorted(wv.vnns.items()):
		vnn_list.append(vnn)

	#vnn_list = [ vnn_list[0] ]#, vnn_list[1], vnn_list[2] ]
	for vnn in vnn_list:
		with tf.name_scope(vnn.name):
			tf.train.import_meta_graph(vnn.meta_filepath)

	weight_address = None
	tensor_weights = tf.trainable_variables()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		time1 = time.time()
		weight_address = sess.run(mod.init_weight(wv.weight_page))
		time2 = time.time()
		print('weight address:', weight_address)
		print('init weight %0.3f ms' % ((time2-time1)*1000.0))

		page_address_list = []
		vnn_no = 0
		for vnn in vnn_list:
			time1 = time.time()
			page_address = sess.run(mod.init_page_table(vnn.weight_page_list))
			time2 = time.time()
			print('[VNN %d] init page table %0.3f ms'
				% (vnn_no, (time2-time1)*1000.0))
			print('page_address:', page_address)
			page_address_list.append(page_address)
			vnn_no += 1

	weight_address = tf.constant(weight_address, name='weight_address')
	page_table_address_list = []
	for i in range(len(page_address_list)):
		page_table_address = tf.constant(page_address_list[i], name='page_table_address/' + str(i))
		page_table_address_list.append(page_table_address)

	get_weight_op_list = []
	for i in range(len(vnn_list)):
		tensor_weights_to_load_list = []
		for weight in tensor_weights:
			if vnn_list[i].name in weight.name:
				tensor_weights_to_load_list.append(weight)
		#print(tensor_weights_to_load_list)
		get_weight_op = mod.get_weight(tensor_weights_to_load_list,
			weight_address, page_table_address_list[i], page_size=wv.weight_per_page)
		get_weight_op_list.append(get_weight_op)

	return vnn_list, get_weight_op_list

def init_virtualization(wv):
	with tf.Graph().as_default() as graph:
		vnn_list = []
		for name, vnn in sorted(wv.vnns.items()):
			vnn_list.append(vnn)
		#vnn_list = [ vnn_list[0] ]#, vnn_list[1], vnn_list[2] ]

		weight_address = None
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			time1 = time.time()
			weight_address = sess.run(mod.init_weight(wv.weight_page))
			time2 = time.time()
			print('weight address:', weight_address)
			print('init weight %0.3f ms' % ((time2-time1)*1000.0))

			page_address_list = []
			vnn_no = 0
			for vnn in vnn_list:
				time1 = time.time()
				page_address = sess.run(mod.init_page_table(vnn.weight_page_list))
				time2 = time.time()
				print('[VNN %d] init page table %0.3f ms'
					% (vnn_no, (time2-time1)*1000.0))
				print('page_address:', page_address)
				page_address_list.append(page_address)
				vnn_no += 1

		page_table_address_list = []
		for i in range(len(page_address_list)):
			page_table_address_list.append(page_address_list[i])

		return vnn_list, weight_address, page_table_address_list

def execute(wv, vnn, weight_address, page_table_address, layers, data_set, label=None):
	print("executing ", vnn.name)

	with tf.Graph().as_default() as graph:
		saver = tf.train.import_meta_graph(vnn.meta_filepath)

		time1 = time.time()
		tensor_weights = tf.trainable_variables()
		time2 = time.time()
		print('tf.trainable_variables %0.3f ms' % ((time2-time1)*1000.0))

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		with tf.Session() as sess:
		#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
		#	graph_options=graph_options, isolate_session_state=True)) as sess:
			get_weight_op = mod.get_weight(tensor_weights,
				weight_address, page_table_address,
				page_size=wv.weight_per_page)

			time1 = time.time()

			a = tf.constant(3)
			b = None


			#print(get_weight_op)
			#print(type(get_weight_op))
			#print(type(sess))
			#print(type(sess._session))
			#tf_session.TF_Run(sess._session, None,
			#	{}, get_weight_op, get_weight_op,
			#	pywrap_tensorflow.TF_NewStatus(), None)

			#tf_session.TF_SessionRun_wrapper(
			#sess._call_tf_sessionrun(
				#sess._session, None, {}, [], [b],
				#None)

			#c = sess.make_callable(get_weight_op)
			#c()
			k = tf.global_variables_initializer()
			#print(type(k))
			#sess._call_tf_sessionrun(None, {}, [], [k.experimental_ref()], None)
			#sess._call_tf_sessionrun(None, {}, [b], [a.op], None)
			#sess._run(None, get_weight_op, None, None, None)
			fetch_handler = session._FetchHandler(
			        sess._graph, get_weight_op, None, feed_handles=None)
			final_fetches = fetch_handler.fetches()
			print('final_fetches')
			print(final_fetches)
			final_targets = fetch_handler.targets()
			print('final_targets')
			print(final_targets)

			fetches = [t._as_tf_output() for t in final_fetches]
			print('fetches')
			print(fetches)
			targets = [op._c_op for op in final_targets]
			print('targets')
			print(targets)

			#results = sess._do_run(None, final_targets, final_fetches,
			#                             {}, None, None)
			sess._call_tf_sessionrun(None, {}, fetches, targets, None)

			#mod.get_weight(tensor_weights,
			#	weight_address, page_table_address,
			#	page_size=wv.weight_per_page).run()
				#options=run_options, run_metadata=run_metadata
			#get_weight_op.run()

			#h = sess.partial_run_setup(get_weight_op)
			#sess.partial_run(h, get_weight_op)
	

			#mod.get_weight(tensor_weights,
			#	weight_address, page_table_address,
			#	page_size=wv.weight_per_page).run()
			time2 = time.time()
			print('get_weight %0.3f ms' % ((time2-time1)*1000.0))

			#time1 = time.time()
			#saver.restore(sess, vnn.model_filepath)
			#time2 = time.time()
			#print('saver.restore %0.3f ms' % ((time2-time1)*1000.0))

			#tl = timeline.Timeline(run_metadata.step_stats)
			#ctf = tl.generate_chrome_trace_format()
			#with open('execute.json', 'w') as f:
			#	f.write(ctf)

			
			keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
			keep_prob = graph.get_tensor_by_name("keep_prob:0")
			x = graph.get_tensor_by_name("neuron_0:0")
			y = graph.get_tensor_by_name("neuron_" + str(layers-1) + ":0")

			data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
			infer_result = sess.run(y, feed_dict={
				x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})

			if label is not None:
				y_ = graph.get_tensor_by_name("y_:0")
				accuracy = graph.get_tensor_by_name("accuracy:0")
				test_accuracy = sess.run(accuracy, feed_dict={
					x: data_set_reshaped, y_: label,
					keep_prob_input: 1.0, keep_prob: 1.0})
				print("Inference accuracy: %f" % test_accuracy)

def main():
	wv = WeightVirtualization()

	vnn_list, weight_address, page_table_address_list = init_virtualization(wv)

	data_list = [ 'mnist_data', 'GSC_v2_data', 'GTSRB_data', 'cifar10_data', 'svhn_data' ]
	layer_list = [ 7, 6, 7, 7, 7 ]

	for i in range(100):
		vnn_no = np.random.randint(len(vnn_list))
		print('vnn_no:', vnn_no)

		data = __import__(data_list[vnn_no])
		data_set = data.test_set()[0]#[0:1]
		label = data.test_set()[1]#[0:1]

		time_s = time.time()
		execute(wv, vnn_list[vnn_no], weight_address, page_table_address_list[vnn_no],
			layer_list[vnn_no], data_set, label)
		time_f = time.time()
		print('total execute %0.3f ms' % ((time_f-time_s)*1000.0))

if __name__ == '__main__':
	main()
