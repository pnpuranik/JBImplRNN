import os
import sys
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
sequenceLength=1000

#function for reading data from all files within a folder and normalizing the data and converting in proper 3d dimensions. Returns data to lstm.py file
def read(folder,testFlag,meanX=-1,varianceX=-1):

	if (testFlag==1):
		print '\n testing files'
	for fileName in os.listdir(folder):
	        filePath=os.path.join(folder,fileName)  #construct file path
	        filex,filey=readFile(filePath)  #dataset from one file
		if (testFlag==1):
			print fileName
		
        #create total dataset from multiple files
	        if 'x' in locals():
	            x=np.vstack([x,filex])  
	            y=np.vstack([y,filey])
	        else:
	            x=filex
	            y=filey
	       
	sess = tf.Session()
	
	#calculate mean and variance of  data if we are in training phase
	if (testFlag==0):
		moment=tf.nn.moments(x,axes=[0])	#parameter wise mean and variance	
		meanX,varianceX=sess.run(moment)
		print 'datatypes'
		print meanX.dtype,varianceX.dtype
	#normalize data
	norm=tf.nn.batch_normalization(x,meanX,varianceX,0,1,1e-3)	
	normX=sess.run(norm)

	#convert data in proper dimensions: 3-d data
	#trainingDataX is total dataset. this will be split into training and validation set in lstm2.py
	trainingDataX=[]
    	i=0	
	while i< len(normX):
		trainingExampleX=[]
        	j=i

		if (i+(sequenceLength))>len(normX):	#check if sequencelength no. of packets are available
			break

		while j<i+sequenceLength:
			trainingExampleX.append(normX[j])	#construct example from the packets
			j=j+1
		
        	i=j
		trainingDataX.append(trainingExampleX)	#construct total 3-d dataset 


#if output per packet is required
	trainingDataY=[]
	i=0	
	while i< len(y):
		trainingExampleY=[]
        	j=i

		if (i+(sequenceLength))>len(y):	#check if sequencelength no. of packets are available
			break

		while j<i+sequenceLength:
			trainingExampleY.append(y[j])	#construct example from the packets
			j=j+1
		
        	i=j
		trainingDataY.append(trainingExampleY)	#construct total 3-d dataset 



#if only one output per sequenceLength packets is required
	#while  (i+(sequenceLength))<= len(y):
	#	trainingExampleY=y[i+(sequenceLength-1)]	#take 1000th packet's output as expected output for a sequence
	#	trainingDataY.append(trainingExampleY)
	#	i=i+sequenceLength
		
	return np.array(trainingDataX),np.array(trainingDataY),meanX,varianceX #return total data




#############################################
#function for reading dat from each file. Skip first few packets info which is invalid. get data in 2d format and return to read function
def readFile(filePath):
    file=genfromtxt(filePath,delimiter=',')
    
    #get necessary columns and skip header (first row)
    #0-currenttime 1- interpacket 2-buffering 3-write 4-source time
    x= file[1:,[0,1,3,4,2]] #skip 0 th row (header)

    #get current threshhold
    y=(file[1:,18])     #skip 0 th row (header)
    y=np.reshape(y,(y.shape[0],1))
    
    #find indices of nonzero elements in the source timestamp column
    nonZeroIndices=np.nonzero(x[:,-1]) 
	 
    #print '{0} nonzero index is '.format(filePath)
    #print nonZeroIndices[0]

    #get all entries starting from first nonzero source time
    x=x[nonZeroIndices[0][0]:,:-1]  #skip source time column
    y=y[nonZeroIndices[0][0]:]

    
    #find delta current time and replace first column with delta values
    xdiff=np.diff(x[:,0])
    diff=np.array([0]) #first entry zero
    x[:,0]=np.hstack([diff,xdiff])  #replace currenttime with delta

    #convert interarrival time in milliseconds
    x[:,1]=np.true_divide(x[:,1],1000)

    returnIndex=(len(x)/sequenceLength)*sequenceLength    #get multiple of sequence length records from each file so that packets from different sessions(files) will not be 													#combined together
    
    return x[:returnIndex],y[:returnIndex]  #return dataset from each file- 2d array with each packet's parameters forming a vector

