# DNN

1. Check the data-file-path before running

2. Run with command "python mainDNN.py"

3. Parameters setting of the network, such as width or depth, can be configured in mainDNN.py

4. Default parameters are as follows. You are strongly suggested to adjust those values properly, since they aren't quite robust.
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

5. Optimizations should be implemented in DNN.py. You can use different activation function or optimizer, such as Adagrad.

6. Due to copyright issue, files in "data" folder are mostly not complete dataset files. You should keep this in mind because it may cause some unexpected error. Of course, you can ignore this if you use your own datasets and parse them on your own. In this case, you might need to implement your own UtilDNN.py.

7. As I said, datasets are not complete. If you use those data for DNN training and testing, results might possibly turn out to be overfitted, and thus the testing accuracy would be quite low.
