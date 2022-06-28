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

	for cl in mylist:
		current_image = cv2.imread(f'{image_dir}/{cl}')
		images.append(current_image)
		student_names.append(os.path.splitext(cl)[0])


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

	# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	# recognizer = cv2.face.LBPHFaceRecognizer_create()
	# current_id = 0
	# label_ids = {}
	# y_labels = []
	# x_train = []

	# for root, dirs, files in os.walk(image_dir):
	# 	for file in files:
	# 		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
	# 			path = os.path.join(root, file)
	# 			label = os.path.basename(root).replace(" ", "-").lower()
	# 			#print(label, path)
	# 			if not label in label_ids:
	# 				label_ids[label] = current_id
	# 				current_id += 1
	# 			id_ = label_ids[label]
	# 			#print(label_ids)
	# 			pil_image = Image.open(path).convert("L")  # grayscale
	# 			size = (200, 200)
	# 			final_image = pil_image.resize(size, Image.ANTIALIAS)
	# 			image_array = np.array(final_image, "uint8")
	# 			#print(image_array)
	# 			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

	# 			for (x, y, w, h) in faces:
	# 				roi = image_array[y:y+h, x:x+w]
	# 				x_train.append(roi)
	# 				y_labels.append(id_)

	# #print(y_labels)
	# #print(x_train)

	# with open("skyhawk/bin/face-labels.pickle", 'wb') as f:
	# 	pickle.dump(label_ids, f)

	# recognizer.train(x_train, np.array(y_labels))
	# recognizer.save("skyhawk/bin/face-trainner.yml")

if __name__ == "__main__":
    Facetrainer()
