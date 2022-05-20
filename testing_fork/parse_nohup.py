import os
import numpy as np
import matplotlib.pyplot as plt
path = 'testing_results/211012_blur_and_clean/'

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

recalls,aps = [],[]
counter = 0
with open(path+"vallog.out", 'r') as f:
	for line in f:
		if '|' in line:
			if  isfloat(line.split("|")[3]):
				print(line)
				#break
				counter+=1
				values = line.split("|")[4:6]
				recall=values[0]
				ap=values[1]
				recalls.append(recall)
				aps.append(ap)

print(counter)

with open(path+'aps.txt', 'w') as f:
	for item in aps:
		f.write("%s\n" % item)

with open(path+'recalls.txt','w') as f:
	for item in recalls:
		f.write("%s\n" %item)
#plt.plot(recalls)
#plt.show()

plt.plot(aps)
plt.show()

'''
import matplotlib
maxcolorrange = 256
l=[]
for row in aps:
	rl=[]
	for i in range(1024):
		pixel = [row[i]/maxcolorrange, row[i+1024]/maxcolorrange, row[i+2048]/maxcolorrange] 
		rl.append(pixel)
	l.append(rl)
matplotlib.imshow(np.array(l)) 
'''
