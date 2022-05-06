#! python3
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler





print("----------------------------------------------------------------")
print("TENSORFLOW TO PREDICT TOTAL EARNINGS")
print("----------------------------------------------------------------")



#===================================
# TF LOGGING MESSAGES
#===================================
# Turn off TensorFlow warning messages in program
# SETTING 2 MINIMISES MESSAGES
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#===================================
# Load training data set from CSV file
#===================================
# USE PANDA TO LOAD DATA
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

#===================================
# GETTING COLOUMNS
#===================================
# TOTAL EARNINGS IS A COLOUMN IN THE SPREAD SHEET
# Pull out columns for X (data to train with) and Y (value to predict)
# AXIS = 1 MEANS COLOUMN AND NOT ROWS
# DROP TO REMOVE TOTAL EARNING COLOUMN
X_training = training_data_df.drop('total_earnings', axis=1)
# ONE X AXIS FOR ALL THE VALUES, THE OTHER FOR TOAL E
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

#===================================
# CREATE GRAPH MODEL FOR TESTING DATA
# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

#===================================
#SCALE SO BIG AND SMALL NUMBERS DIFFERENCE ISN'T TOO LARGE
#===================================
# FOR TRAINING DATA
# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))


# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

#===================================
# FOR TESTING DATA AS WELL
#===================================
# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)



#===================================
# PARAMETERS
# Define model parameters
learning_rate = 0.001
# 1 TRAINING LOOP = EPOCH = ONE FULL TRAINING PASS. 100 TRAINING LOOPS
training_epochs = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50


# Section One: Define the layers of the neural network itself

# Input Layer
#GIES PREFIX OF INPUT ALLOWING IT TO GENERATE DIAGRAMS
# EVERYTIME IT RUNS IT NEEDS A NEW PLACEHOLDER
# SHAPE = NONE TO ACCEPT BATCHES OF ANY SIZE
#NUMBER OF INPUTS DEFINED AS 9 ALREADY
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))


#===================================
#9 INPUTS -> LAYER 1 -> LAYER 2 -> LAYER 3
#===================================
# Layer 1
# WEIGHT CONNECTION
# BIAS = 1
# SHAPE SHOULD BE SAME AS NUMBER OF NODES SO LAYER 1 NODES
# SHAPE => ON LEFT IS INPUTS ON RIGHT IS NODES
# INITIALISER BES INITIAL VALUES ALGO IS USING XAVIER TO CALCULATE INITIAL VALUES FOR ALL INITIALISERS

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    # OUPUT MATRIX MULTIPLIER. MULPIT INPUTS TO RATE AND ADD BIAS. THIS IS HOW YOU DEFINE A FULLY CONNECTING LAYER
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# REPEAT WITH THE OTHER LAYERS
# LYAER 1 NODES ON LEFT
# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    # MATRIX CALCUATION WITH LAYER 1 OUTPUT
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# DEFINE FINAL LAYER
# Output Layer WITH VAR number_of_outputs



#===================================
#COST FUNCTION
#===================================
# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training
# TO TRAIN IT WE NEED LOSS FUNCTION.
# REFER TO DIAGRAM ON TABLET
# TELLS HWO WRONG THE NEURAL NETWORK IS TO TRAIN IT
with tf.variable_scope('cost'):
    # NEW VALUE FEED INTO EACH TIME SO PLACEHOLDER NODE
    # SHAPE = ONE AND 1 IS BECAUSE ONE OUPUT
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    # TO CALCULATE COST WE MEAN SQUARRE ERROR. PASS IN PREDICTION AND COMPARE TO EXPECTED VALUE
    ## PASS IN PREDICTION AND EXPTECTED VALUE WHICH IS Y
    # WRAP WITH REDUCE TO GET AVERAGE VALUE OF MEAN SQUARE ERROR
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))





#===================================
# CALL OPTIMISER
#===================================
# Section Three: Define the optimizer function that will be run to optimize the neural network
# CREATE OPTIMISER BY USING THE ADAM OPTIMISER BY PASSING IN COST
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)






#=================================
# CREATING GRPAHS
#=================================
# CAN ALSO SUMMARY.AUDIO TO SEE AUDIO .IMAGES FOR IMAGES
# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    tf.summary.histogram('predicted_value', prediction)
    summary = tf.summary.merge_all()

#=================================






#=================================
# CREATING SESSION
#=================================
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    #===================================
    # RUN ENTIRE SESSION
    #===================================
    #RUNNING ENTIRE SESSION
    # PSVM 
    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    #===================================
    # END
    #===================================




    #===================================
    # LOADING FROM MODEL SAVED IN LOGS
    #===================================
    # COMMENT OUT 'RUN ENTIRE SESSION' ABOVE
    # When loading from a checkpoint, don't initialize the variables!
    # COMMENT OUT THIS LINE
    # session.run(tf.global_variables_initializer())
    # Instead, load them from disk:
    #UNCOMMENT THIS TO LOAD MODEL INSTEAD
    #===================================
    # USE PREVIOUS TRAINED MODEL
    #===================================
    # saver.restore(session, "logs/trained_model.ckpt")
    # print("Trained model loaded from disk.")
    #===================================
    # END
    #===================================





    #===================================
    # CREATE DIFFERENT MODEL LOG FILES EVERY TIME
    #===================================
    # PREVENTS OVERWRITING
    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter("./logs/{}/training".format(RUN_NAME), session.graph)
    testing_writer = tf.summary.FileWriter("./logs/{}/testing".format(RUN_NAME), session.graph)
    #===================================
    # END
    #===================================


    

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        #===================================
        # LOGGING AFTER EVERY EPOCHS
        #===================================
        # Every few training steps (EPOCHs), log our progress
        if epoch % 5 == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            # GETS CURRENT
            # GET TRAINING COST FOR EVEY 5 EPOCJ LOOP
            # CALL THE COST FUNCTION
            # ADD IN , SUMMARY TO GET SUMMARY INFO AS WELL IN ONE GO
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

    # Training is now complete!



    #===================================
    # FINAL COST
    #===================================
    # PRINT OUT FINAL COST TO SEE THE VALUES OF TRAINING
    #COST WILL DECREASE OVER TIME
    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))


    #===================================
    # LOAD UP TENSORBOARD FOR DIAGRAMS
    #===================================
    # TYPE IN TERMINAL
    # 05/logs IS THE DIRECTORY
    # tensorboard --logdir=05/logs

    
