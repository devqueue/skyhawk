import cv2
import os
import numpy as np
from PIL import Image
import pickle
import face_recognition



def Facetrainer():
	'''Train skyhawk classifier on captured data'''
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	direc = "../facedata"
	bin_direc = "../bin"
	image_dir = os.path.join(BASE_DIR, direc)
	bin_dir = os.path.join(BASE_DIR, bin_direc)


	images = []
	student_names = []
	mylist = os.listdir(image_dir)

	for image in mylist:
		current_image = cv2.imread(f'{image_dir}/{image}')
		images.append(current_image)
		student_names.append(os.path.splitext(image)[0])


	#get encodings of known faces 
	def getEndcordings(images):
		encode_list = []
		for img in images:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			encode = face_recognition.face_encodings(img)[0]
			encode_list.append(encode)
		return encode_list


	know_encodelist = getEndcordings(images)
	know_encodelist= np.array(know_encodelist)

	info_dict = {}

	for name,encordings in zip(student_names,know_encodelist):
		info_dict[name] = encordings

	bin_file = os.path.join(bin_dir, 'facedata.bin')

	with open(bin_file,'wb') as py:
		pickle.dump(info_dict,py)

	print("Training completed")


if __name__ == "__main__":
	Facetrainer()
