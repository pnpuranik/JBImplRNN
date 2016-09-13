import numpy as np
import tensorflow as tf
import sys
from time import sleep
import time
from preprocessing import read

#this file contains code for creating a lstm network with one hidden layer.
#it trains the network, validates it periodically and tests it at the end

#function for shuffling dataset	
def shuffleInUnison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
	


#Execution starts here
startTime=time.time()
print "started at {0}".format(startTime) 

sequenceLength=1000	#no of previous packets to be considered
alpha=float(sys.argv[2])	#0.05 works good for RMSprop
epoch=20	#no. of times dataset will be used for training
numOfHiddenUnits=int(sys.argv[3])	#8 works good for RMSprop 

#display all values
print "\nsequenceLength ",sequenceLength
print 'alpha ',alpha
print 'epoch ',epoch
print 'numOfHiddenUnits ',numOfHiddenUnits
batchSize = 100

#file to write validation results
testFile=open('testoutput.txt','a+')

print sys.argv[1]
normX,normY,meanX,varianceX=read(sys.argv[1],0)	#send folder name of qos files and get normalized data from all the files in that folder 


mean=tf.Variable(meanX,dtype=tf.float32)
variance=tf.Variable(varianceX,dtype=tf.float32)


print "\nTotal data size"
print normX.shape, normY.shape

totalDataLength=len(normX)

sess = tf.Session()

trainIndex=(totalDataLength*70)/100					#split data into training and validation/testing sets

#split x into training and testing data
print "splitting trainIndex ",trainIndex

trainDataX=normX[:trainIndex]
trainDataY=normY[:trainIndex]

testDataX=normX[trainIndex:]
testDataY=normY[trainIndex:]

					#parameters-___,sequence length, Input parameters per packet
data=tf.placeholder(tf.float32,[None,sequenceLength,4])
target = tf.placeholder(tf.float32, [None, sequenceLength,1])			#parameters-___,sequence length, output parameters per packet

#create 2nd layer - LSTM 
cell = tf.nn.rnn_cell.LSTMCell(numOfHiddenUnits)

#calculate hidden layer outputs
hiddenVals,cellState = tf.nn.dynamic_rnn(cell, data,dtype=tf.float32)

#weight and bias for the hiddden to output layer
weight = tf.Variable(tf.truncated_normal([numOfHiddenUnits, int(target.get_shape()[2])]), name='weights')
bias = tf.Variable(tf.constant(0.1, shape=[  int(target.get_shape()[2])      ]), name='biases')


#convert 3d hiddenVals to 2d in order to carry out matrix-multiplication and convert the output back to 3d format 
hiddenVals=tf.reshape(hiddenVals,[-1,numOfHiddenUnits])
predictions=tf.matmul(hiddenVals,weight)+bias

predictions=tf.reshape(predictions,[  -1, int(target.get_shape()[1]),  int(target.get_shape()[2]) ])

#Root mean squared error- cost function
cost = tf.reduce_mean( ( tf.reduce_sum(tf.square(target-predictions),reduction_indices=[1]) ) )		#minimize along corresponding columns of different sequences
tf.scalar_summary('cost',cost)		#for creating tensorboard graph

#train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)	#0.0003
train_step = tf.train.RMSPropOptimizer(alpha).minimize(cost)		#0.03 or more

init_op = tf.initialize_all_variables()
saver=tf.train.Saver(tf.all_variables())

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('/home/rrewale/train',sess.graph)
test_writer = tf.train.SummaryWriter('/home/rrewale/test')

sess.run(init_op)

print sess.run(mean)
print sess.run(variance)

#divide dataset into batches for faster convergence
no_of_batches = np.true_divide(len(trainDataX),batchSize)
no_of_batches=int(no_of_batches)
print 
print 'No of batches ',no_of_batches
print 

#testDataX,testDataY=shuffleInUnison(testDataX,testDataY)	#shuffle testing data for better testing

for i in range(epoch):
    ptr = 0
   # trainDataX,trainDataY=shuffleInUnison(trainDataX,trainDataY)	#shuffle data during each epoch fro training
    avgError=0	#calculate avg error for the whole epoch

    for j in range(no_of_batches):
        inp, out = trainDataX[ptr:ptr+batchSize], trainDataY[ptr:ptr+batchSize]

        ptr+=batchSize
	err= sess.run(cost,feed_dict={data: inp, target: out})	#error for each batch
	avgError+=err
	trainSummary,traincost=sess.run([merged,cost], feed_dict={data: inp, target: out})
	
	train_writer.add_summary(trainSummary, i*no_of_batches+j)		#add error for each batch to graph
        sess.run(train_step,{data: inp, target: out})	#train the NN
	print "Error for batch- ",str(j)," - ",str(err)
	
    print "\nEpoch - ",str(i)
    print "avg error for epoch {0}".format(np.true_divide(avgError,no_of_batches))

#validation  after few epochs to check convergence
    if i%5==0:
	err= sess.run(cost,feed_dict={data: testDataX, target: testDataY})
	testPred=sess.run(predictions,feed_dict={data: testDataX, target: testDataY})
	testSummary,testcost=sess.run([merged,cost], feed_dict={data: testDataX, target: testDataY})
	test_writer.add_summary(testSummary, i)		#add testing error to graph

        testFile.write('\nError for whole testing data '+str(i)+' '+str(err)+'\n')

	#print output for few tesing examples for manual comparison 
	testFile.write('\n\nOutput for 40\n')
	testFile.write(str(testPred[:40]))

testFile.write('\n\nExpected output for40\n')
testFile.write( str(testDataY[:40]))
testFile.write('\n\n')
	
testFile.write('\nPredicted outputs for 40\n')
testFile.write(str(testPred[:40]))
#np.savetxt(testFile,testPred[:40])
testFile.write('\n')

endTime=time.time()

perErrors=np.true_divide( (testPred-testDataY) , testDataY)
perError=np.mean(np.absolute(perErrors))

testFile.write("\nMean Percentage error ")
testFile.write(str(perError))

print "Endtime if {0}".format(endTime)
print "Total time taken {0}".format(endTime-startTime)

#save model
path="/home/rrewale/Desktop/Shared/rnn/lstmseqtoseq/OUTPUTS/"
savePath=saver.save(sess,path+"model.ckpt")
saver.save(sess,'seqtoseqmodel')

#check if directory for testing is given. if yes the execute this code, otherwise don't
if (len(sys.argv)==5):
							#1 indicates testing
	testingX,testingY,meanX,varianceX=read(sys.argv[4],1,meanX,varianceX)
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
