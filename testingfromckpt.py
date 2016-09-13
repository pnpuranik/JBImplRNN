import numpy as np
import tensorflow as tf
import sys
import time
from preprocessing import read

sequenceLength=1000
numOfHiddenUnits=8
data=tf.placeholder(tf.float32,[None,sequenceLength,4])
target = tf.placeholder(tf.float32, [None, sequenceLength,1])		


cell = tf.nn.rnn_cell.LSTMCell(numOfHiddenUnits)

hiddenVals,cellState = tf.nn.dynamic_rnn(cell, data,dtype=tf.float32)

#weight and bias for the hiddden to output layer
weight = tf.Variable(tf.truncated_normal([numOfHiddenUnits, int(target.get_shape()[2])]), name='weights')
bias = tf.Variable(tf.constant(0.1, shape=[  int(target.get_shape()[2])      ]), name='biases')

meanX=tf.cast( tf.Variable(tf.zeros([4]), dtype=tf.float32), dtype=tf.float64)
varianceX= tf.cast( tf.Variable(tf.zeros([4]), dtype=tf.float32 ) , dtype=tf.float64)

hiddenVals=tf.reshape(hiddenVals,[-1,numOfHiddenUnits])
predictions=tf.matmul(hiddenVals,weight)+bias

predictions=tf.reshape(predictions,[  -1, int(target.get_shape()[1]),  int(target.get_shape()[2]) ])

sess=tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

saver=tf.train.Saver(tf.all_variables())

path="/home/rrewale/Desktop/Shared/rnn/lstmseqtoseq/OUTPUTS/"
saver.restore(sess,path+"model.ckpt")
mean=sess.run(meanX)
variance=sess.run(varianceX)

testingX,testingY,meanX,varianceX=read(sys.argv[1],1,mean,variance)	#read data for testing 

testPred=sess.run(predictions,feed_dict={data: testingX})

#dump outputs of testing to files 

#predicted outputs
fileNo=0
for dslice in testPred:	
	fileNo=fileNo+1	
	np.savetxt(path+'testpred'+str(fileNo)+'.csv',dslice, delimiter=",",fmt='%10.5f')	

#expected outputs
fileNo=0	
for dslice in testingY:	
	fileNo=fileNo+1	
	np.savetxt(path+'testexp'+str(fileNo)+'.csv',dslice, delimiter=",",fmt='%10.5f')

sess.close()
