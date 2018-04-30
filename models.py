import os
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add
from keras.layers import BatchNormalization, Reshape
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adamax
import argparse


def FCN_Vgg16(input_shape=None, mode='fcn32s',n_classes=7):
	img_input = Input(shape=input_shape)

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
	block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
	block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
	block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

	# Block 5
	block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
	block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
	block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
	block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

	# Convolutional layers transfered from fully-connected layers
	fc_conv1 = Conv2D(4096, (3, 3), activation='relu', padding='same', name='fc_conv1')(block5_pool)
	#fc_conv1 = Dropout(0.3)(fc_conv1)
	
	fc_conv2 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc_conv2')(fc_conv1)
	#fc_conv2 = Dropout(0.3)(fc_conv2)
	
	#classifying layer
	fc_conv3 = Conv2D(n_classes, (1, 1),kernel_initializer='he_normal', activation='relu', padding='same', strides=(1, 1))(fc_conv2)

	if(mode == 'fcn32s'):
		o = Conv2DTranspose(filters=n_classes, kernel_size=(64,64), strides=(32,32), use_bias=False, padding='same')(fc_conv3)

	elif(mode == 'fcn16s'):
		up_1 = Conv2DTranspose(filters=n_classes,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fc_conv3)
		block4_fc =  Conv2D(n_classes, (1, 1),kernel_initializer='he_normal', activation='relu', padding='same', strides=(1, 1))(block4_pool)
		fuse_layer = Add()([up_1,block4_fc])
		o = Conv2DTranspose(filters=n_classes, kernel_size=(32,32),strides=(16,16),use_bias=False, padding='same')(fuse_layer)

	elif(mode == 'fcn8s'):
		up_1 = Conv2DTranspose(filters=n_classes, kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fc_conv3)
		block4_fc =  Conv2D(n_classes, (1, 1),kernel_initializer='he_normal', activation='relu', padding='same', strides=(1, 1))(block4_pool)
		fuse_layer = Add()([up_1,block4_fc])
		
		up_2 = Conv2DTranspose(filters=n_classes, kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fuse_layer)
		block3_fc = Conv2D(n_classes, (1, 1),kernel_initializer='he_normal', activation='relu', padding='same', strides=(1, 1))(block3_pool)
		o = Add()([up_2,block3_fc])
		o = Conv2DTranspose(filters=n_classes, kernel_size=(16,16),strides=(8,8),use_bias=False, padding='same')(o)
	
	o = Activation('softmax')(o)
	model = Model(img_input, o)

	weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	model.load_weights(weights_path, by_name=True)

	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-mo', '--mode', help='training mode', type=str)
	args = parser.parse_args()
	model = FCN_Vgg16(input_shape=(512,512,3), mode=args.mode)
	model.summary()
	