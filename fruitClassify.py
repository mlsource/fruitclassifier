import tensorflow as tf, sys
import os
from os import listdir
from os.path import isfile, join

#load the learned model

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in  tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


# get the test director under which the category sub directories #are listed
testImage_path = sys.argv[1]

subdirectories = [subdir for subdir in listdir(testImage_path)]

for d in subdirectories:
	file_dir_path = join(testImage_path, d)
	onlyfiles = [f for f in listdir(file_dir_path) if isfile(join(file_dir_path, f))]
	#print(onlyfiles);
	for f in onlyfiles:
		# Read in the image_data
		image_path = join(file_dir_path, f)
		print(image_path)
		image_data = tf.gfile.FastGFile(image_path, 'rb').read()

		with tf.Session() as sess:
    			# Feed the image_data as input to the graph and get first prediction
    			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    			predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
    			# Sort to show labels of first prediction in order of confidence
    			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
			for node_id in top_k:
        			human_string = label_lines[node_id]
        			score = predictions[0][node_id]
        			print('%s (score = %.5f)' % (human_string, score))
