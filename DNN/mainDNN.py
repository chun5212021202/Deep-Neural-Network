
import DNN
import UtilDNN
import time



start_time = time.time()
#########################################################
#														#
#						GET START						#
#														#
#########################################################


DNN_EPOCH = 100
DNN_WIDTH = 128
DNN_DEPTH = 2
DNN_X_DIMENSION = 69
DNN_Y_DIMENSION = 48
DNN_LEARNING_RATE = 0.001
DNN_BATCH_SIZE = 128
DNN_MOMENTUM = 0.9
DNN_DECAY = 0.999999
DNN_OUTPUT_FILE = 'result.lab'





dnn = DNN.DNN( DNN_X_DIMENSION, DNN_Y_DIMENSION, DNN_DEPTH, DNN_WIDTH, DNN_LEARNING_RATE, DNN_BATCH_SIZE, DNN_MOMENTUM, DNN_DECAY )

data = UtilDNN.LoadTrainDNN( 'data/train_light.ark','data/train.lab' )

UtilDNN.TrainDNN( data, dnn, DNN_BATCH_SIZE, DNN_EPOCH )

UtilDNN.TestDNN( 'MLDS_HW1_RELEASE_v1/fbank/test_light.ark', dnn, DNN_BATCH_SIZE, DNN_OUTPUT_FILE )	#change back to test.ark!!!!!!!!!!!!!



print 'Learning Rate at last : %r' % (dnn.learning_rate.get_value())
print("--- DNN %s seconds ---" % (time.time() - start_time))
print



trigger = raw_input("Do You Want To Check The Answer ? (Y/N): ")
if trigger == 'Y' or 'y' :
	UtilDNN.CheckDNN(DNN_OUTPUT_FILE, 'MLDS_HW1_RELEASE_v1/label/test_light.lab')





