import os
from os.path import isfile, isdir, join
from multiprocessing import freeze_support
import concurrent.futures
from PIL import Image
import shutil
import random
import tensorflow as tf
from datetime import datetime, timedelta
import numpy as np

def extract_luma(img_path):
	img = Image.open(img_path)

	noAA_image = img.split()

	r = np.float32(noAA_image[0])
	g = np.float32(noAA_image[1])
	b = np.float32(noAA_image[2])
	y = r * 0.299 + g * 0.587 + b * 0.114
	
	noAA_tensor_luma = y.reshape(noAA_image[0].size[1], noAA_image[0].size[0], 1) / 255

	img.close()

	return noAA_tensor_luma

class NnaaDataset(tf.keras.utils.PyDataset):
	def __init__(self, bases_dir, targets_dir, batch_size, use_cache = False, **kwargs):
		super().__init__(**kwargs)
		self.img_names = list(set(os.listdir(bases_dir)).union(os.listdir(targets_dir)))
		self.img_names = [f for f in self.img_names if isfile(join(bases_dir, f)) and isfile(join(targets_dir, f))]

		random.shuffle(self.img_names)

		self.bases_dir = bases_dir
		self.targets_dir = targets_dir
		self.batch_size = batch_size

		self.cache_built = False

		if use_cache:
			self.cache_tensors = []
			for i in range(len(self)):
				self.cache_tensors.append(self[i])
		
			self.cache_built = True


	def __len__(self):
		return (len(self.img_names) // self.batch_size)

	def __getitem__(self, idx):

		if self.cache_built:
			return self.cache_tensors[idx]

		inputs = []
		targets = []

		r_idx = idx * self.batch_size

		for i in range(r_idx, r_idx + self.batch_size):
			
			img_name = self.img_names[i]

			base_path = join(self.bases_dir, img_name)
			target_path = join(self.targets_dir, img_name)
			
			x = extract_luma(base_path)
			y = extract_luma(target_path)

			inputs.append(x)
			targets.append(y - x)

		return (np.half(inputs), np.half(targets))
	


if __name__ == "__main__":
	freeze_support()

	# base_dir directory are the images without AA and the target_dir directory are the images with AA, they are matched by names
	base_dir_path = "data/train/bad/1280x720"
	target_dir_path = "data/train/fixed/1280x720"

	base_dir_path_test = "data/test/bad/2560x1440"
	target_dir_path_test = "data/test/fixed/2560x1440"

	model_name = "nnaa"
	models_path = ".."

	lr = 0.00001
	dropout_rate = 0

	input = tf.keras.Input(shape=(None, None, 1), name="img")

	x = tf.keras.layers.PReLU(shared_axes=[1, 2])(tf.keras.layers.Conv2D(32, 8, strides=2, padding='same')(input))  
	x = tf.keras.layers.PReLU(shared_axes=[1, 2])(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
	x = tf.keras.layers.PReLU(shared_axes=[1, 2])(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
	x = tf.keras.layers.PReLU(shared_axes=[1, 2])(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(x))
	output = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding='same', name='conv2d_final')(x)

	model_directory = join(models_path, model_name) 

	if(not isdir(model_directory)):
		os.mkdir(model_directory)	

	model_path = join(model_directory, model_name) + ".keras"

	loss_fn = tf.keras.losses.MeanSquaredError()

	if(isfile(model_path)):
		print("loading model")
		model = tf.keras.models.load_model(model_path)
	else:
		print("create model")
		model = tf.keras.Model(input, output, name=model_name)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
				loss=loss_fn,
				metrics=['mean_squared_error'])

	model.summary()

	if(model.optimizer.learning_rate != lr):
		model.optimizer.learning_rate = lr
		print("Learning rate changed")
		
	print(f"Learning rate : {lr}")

	train_dataset = NnaaDataset(base_dir_path, target_dir_path, 16, use_cache=True)
	test_dataset = NnaaDataset(base_dir_path_test, target_dir_path_test, 4, use_cache=True)

	best_error_value = float('inf')
	if(isfile(join(model_directory, "bestError.npy"))):
		best_error_value = np.load(join(model_directory, "bestError.npy")).item()
		
	print(f"Best Error Value : {best_error_value}") 

	i = 0

	while True:
		i += 1
		
		print(" ")
		print(f"Run {i}")
		
		model.fit(train_dataset, epochs=5)
		
		eval = model.evaluate(test_dataset, verbose=2)
		
		if(eval[0] < best_error_value):
			best_error_value = eval[0]
			np.save(join(model_directory, "bestError"), best_error_value)
			model.save(model_path)


