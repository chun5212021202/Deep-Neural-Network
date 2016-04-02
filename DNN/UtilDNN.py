
from random import shuffle


#########################################################
#														#
#						FUNCTIONS						#
#														#
#########################################################
def LoadTrainDNN(training_file_path, state_file_path) :
	#	read file
	print 'File \'train.ark\' Loading.....'
	training_file = open(training_file_path,'r')
	training_data_list = training_file.read().splitlines()


	#	read file
	print 'File \'train.lab\' Loading.....'
	state_file = open(state_file_path, 'r')
	state_file_list = state_file.read().splitlines()


	#	read file
	print 'File \'48_39.map\' Loading.....'
	map_file = open('data/48_39.map','r')
	map_list = map_file.read().splitlines()


	#
	print 'Parsing Files.....'	
	training_data = {}
	for line in training_data_list :
		line = line.strip()
		training_data[line.split()[0]] = [float(data) for data in line.split()[1:]]
	key =  list(training_data.keys())
	

	#
	state_data = {}
	for line in state_file_list :
		line = line.strip()

		state_data[line.rsplit(',', 1)[0]] = line.rsplit(',', 1)[1]

	#
	map_data =  {}
	for idx,line in enumerate(map_list) :
		line = line.strip()
		map_data[line.split()[0]] = idx	

	return key, training_data, state_data, map_data



def TrainDNN(data, dnn_object ,batch_size, epoch) :	# train one epoch
	key = data[0]
	training_data = data[1]
	state_data = data[2]
	map_data = data[3]

	
	print ' *** START TRAINING *** '
	total_cost = 0


	for turns in range(epoch) :
		print 'EPOCH %d ' % (turns)
		count = 0
		shuffle(key)

		training_batch_x=[]
		training_batch_y=[]
		for instance_id in key :	
			training_batch_x.append( training_data[instance_id] )

			training_data_y = [0]*48
			training_data_y[ map_data[state_data[instance_id]] ] = 1
			training_batch_y.append( training_data_y )
			if count % batch_size == batch_size-1 :

				print 'EPOCH %d ; Batch Number : %d' % (turns, count//batch_size)
				temp = dnn_object.train( training_batch_x, training_batch_y )
				print temp
				print

				training_batch_x=[]
				training_batch_y=[]

				if turns == epoch-1 :
					total_cost = total_cost+temp[0]
				
			count = count+1
		#return
	print 'Total Cost of Last Epoch : %f' % (total_cost)
	print

	#print count



def TestDNN(testing_file_path, dnn_object, batch_size, output_file) :

	print 'File \'48_39.map\' Loading.....'
	map_file = open('data/48_39.map','r')
	map_list = map_file.read().splitlines()
	map_data =  {}

	for idx,line in enumerate(map_list) :
		line = line.strip()
		map_data[idx] = line.split()[0]

	#
	print 'File \'test.ark\' Loading.....'
	testing_file = open (testing_file_path,'r')
	testing_data_list = testing_file.read().splitlines()
	testing_data = {}

	print ' *** START TESTING *** '
	testing_batch = []
	testing_id = []
	testing_predict = []
	for idx,line in enumerate(testing_data_list) :
		line = line.strip()

		testing_data = [float(data) for data in line.split()[1:]]
		testing_batch.append(testing_data)
		testing_id.append(line.split()[0])

		if idx % batch_size == batch_size-1 :
			#print 'Test Number : %d' % (idx//batch_size)
			testing_predict += dnn_object.test(testing_batch).tolist() 	# array of number (batchsize)
			testing_batch = []

	#
	
	result_file = open(output_file, 'w+')
	for idx,pridiction_num in enumerate(testing_predict) :

		result_file.write(testing_id[idx] + ',' + map_data[pridiction_num] + ',' + str(pridiction_num) + '\n')


def CheckDNN(result_file_path, answer_file_path) :
	count = 0

	print 'File \'48_39.map\' Loading.....'
	map_file = open('data/48_39.map','r')
	map_list = map_file.read().splitlines()
	map_data =  {}

	for line in map_list :
		line = line.strip()
		map_data[line.split()[1]] = line.split()[0]	# 39->48



	result_file = open(result_file_path)
	result_data_list = result_file.read().splitlines()
	result_data = {}
	for line in result_data_list :
		line = line.strip()
		result_data[line.rsplit(',')[0]] = line.rsplit(',')[1]


	answer_file = open(answer_file_path)
	answer_data_list = answer_file.read().splitlines()
	answer_data = {}
	for line in answer_data_list :
		answer_data[line.rsplit(',',1)[0]] = map_data[line.rsplit(',',1)[1]]

	key =  list(result_data.keys())

	for instance_id in key :
		if result_data[instance_id] == answer_data[instance_id] :
			count += 1
	percentage = float(count)/len(key)

	print "Correct Answers : %d" % (count)
	print "Total Datas : %d" % (len(key))
	print "Correct Rate : %r" % (percentage)
