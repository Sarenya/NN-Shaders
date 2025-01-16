from PIL import Image
import numpy as np
import os
import sys
import tensorflow as tf

def save(model, path_img, with_diff):
	
	path_no_ext = path_img[0:path_img.rfind('.')]
	noAA_image = Image.open(path_img).split()
	
	r = np.float32(noAA_image[0])
	g = np.float32(noAA_image[1])
	b = np.float32(noAA_image[2])
	y = r * 0.299 + g * 0.587 + b * 0.114
	cb = r * -0.1687 + g * -0.3313 + b * 0.5
	cr = r * 0.5 + g * -0.4187 + b * -0.0813
	
	noAA_img_tensor = y.reshape(1, noAA_image[0].size[1], noAA_image[0].size[0], 1)
	noAA_img_tensor /= 255
	
	maybe_better_image = model(noAA_img_tensor)
	
	noAA_img_tensor += maybe_better_image
	
	maybe_better_image_y = np.float32(noAA_img_tensor).reshape(noAA_image[0].size[1], noAA_image[0].size[0])
	maybe_better_image_y = (maybe_better_image_y * 255).round().clip(0, 255)
	
	r = Image.fromarray(np.uint8((maybe_better_image_y + 1.402 * cr).round().clip(0, 255)))
	g = Image.fromarray(np.uint8((maybe_better_image_y - 0.34414 * cb - 0.71414 * cr).round().clip(0, 255)))
	b = Image.fromarray(np.uint8((maybe_better_image_y + 1.772 * cb).round().clip(0, 255)))
	
	Image.merge('RGB', [r, g, b]).convert('RGB').save(path_no_ext + '_AA.png')
	
	if(with_diff):
		diff = np.absolute(np.array(maybe_better_image).reshape(noAA_image[0].size[1], noAA_image[0].size[0])) * 255
		diff = diff.round().clip(0, 255)
		diff = Image.fromarray(np.uint8(diff))
		Image.merge('RGB', [diff, diff, diff]).save(path_no_ext + '_AA_black_diff.png')


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python use.py image.png")
	else:
		model = tf.keras.models.load_model('nnaa.keras')
		
		save(model, sys.argv[1], True)
	
	