import os
import sys
import shutil
import subprocess

alpha=[0.05]
hiddenunits=[8]	

for i in alpha:
	for j in hiddenunits:
		command='python lstm2.py /home/rrewale/Desktop/Shared/rnn/CompleteData/ {0} {1}  > trainoutput.txt'.format(i,j)
		
		#execute the network code
		os.system(command)

		path='OUTPUTS/'+str(i)+"_"+str(j)
	
		if not os.path.exists(path):
			os.mkdir(path)

		shutil.move("trainoutput.txt",path+"/trainoutput.txt")
		shutil.move("testoutput.txt",path+"/testoutput.txt")

		folder='/home/rrewale/train'
		for fileName in os.listdir('/home/rrewale/train'):
			filePath=os.path.join(folder,fileName)
			shutil.move(filePath,path+"/"+fileName)
		
		folder='/home/rrewale/test'
		for fileName in os.listdir('/home/rrewale/test'):
			filePath=os.path.join(folder,fileName)
			shutil.move(filePath,path+"/"+fileName)
		
