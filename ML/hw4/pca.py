import numpy as np 
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
face_path = os.path.join(base_path,'face')
img = []
size = 64
subject = ['A','B','C','D','E','F','G','H','I','J']
#num = ['00','01','02','03','04','05','06','07','08','09']
num = [n for n in range(10)]
for file in os.listdir(face_path):
	if file[0] in subject and file[1] == '0' and int(file[2]) in num:
		img.append(np.reshape(np.array(Image.open(os.path.join(face_path,file))),(size*size)))
		
img = np.array(img)
img_mean = img.mean(axis = 0,keepdims = True)
img_ctr = img - img_mean
#plot average face
'''
plt.figure()
plt.imshow(img_mean.reshape(size,size),cmap = 'gray')
plt.colorbar()
plt.tight_layout()
fig.gcf()
plt.show()
#fig.savefig('average_face.png',dpi = 200,format = 'png')
'''
#plot eigenface
u, s, v = np.linalg.svd(img_ctr)
eigenface = np.array([v[i] for i in range(9)])
'''
fig = plt.figure()
for i,face in enumerate(eigenface):
	ax = fig.add_subplot(3,3,i+1)
	ax.imshow(face.reshape(64,64),cmap = 'gray')
	plt.tight_layout()
plt.show()
'''
#plot reconstruction
eigen_size = 60
img_reduced = np.array( [np.dot(img_ctr,np.array(v[i])) for i in range(eigen_size)] )

img_reconstruct = 0
for i in range(eigen_size):
	img_reconstruct += img_reduced[i].reshape(img_reduced[i].shape[0],1)*v[i]
img_reconstruct += img_mean
'''
fig = plt.figure(figsize=(18,18))
fig.suptitle("Reconstructed faces", fontsize=20)
for i,face in enumerate(img_reconstruct):
	ax = fig.add_subplot(10,10,i+1)
	ax.imshow(face.reshape(64,64),cmap = 'gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.savefig('original.png',dpi = 200,format = 'png')
'''

def RMSE(x_original,x_reconstruct):
	diff = np.abs(x_original/255-x_reconstruct/255)
	return np.sqrt(np.sum(diff*diff)/(size*size)/100)

print(RMSE(img,img_reconstruct))
