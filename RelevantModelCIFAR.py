import keras
from keras.preprocessing import image
from keras import backend, layers
from keras.models import Model 
from keras.layers import Input, Dense
import math
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D




def PlainNets(depth=20):
     
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 56 or 110 in [b])')    
    n = math.floor((depth-2)/6)
    print("value of n is", n)
    model = keras.models.Sequential()
    model.add(Conv2D(16,(3,3), strides=(1, 1), padding='same', kernel_initializer='he_normal',name='block1_conv1', input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    
    for i in range(2*n):
        conv_name_base = 'plain_' + str(depth) + 'block1' + '_branch' + str(i)
        model.add(Conv2D(16,(3,3), padding='same',kernel_initializer='he_normal', name=conv_name_base))
        model.add(BatchNormalization())
        model.add(Activation('relu')) 
    
    model.add(Conv2D(32, (3,3), padding='same', strides=(2,2),  kernel_initializer='he_normal',name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for i in range(2*n-1):
        conv_name_base = 'plain_' + str(depth) + 'block2' + '_branch' + str(i)
        model.add(Conv2D(32,(3,3), padding='same', kernel_initializer='he_normal',name=conv_name_base))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Conv2D(64, (3,3), padding='same', strides=(2,2),  name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for i in range(2*n-1):
        conv_name_base = 'plain_' + str(depth) + 'block3' + '_branch' + str(i)
        model.add(Conv2D(64,(3,3), padding='same', kernel_initializer='he_normal',name=conv_name_base))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
   
    model.add(GlobalAveragePooling2D(name='block3_pool')) 
    model.add(Dense(10, activation='softmax', name='fc1'))
    
    return model





def identity_block(input_tensor, filters, strides = (1,1)):
    
    
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    filters1, filters2 = filters
    stride1, stride2 = strides 
    
    x = layers.Conv2D(filters1, (3, 3),
                      kernel_initializer='he_normal',
                      padding = 'same')(input_tensor)
    
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, (3,3),
                      kernel_initializer='he_normal',
                      padding = 'same')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, filters, strides = (2,2)):
    
    
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    filters1, filters2 = filters
    stride1, stride2 = strides 
    X_shortcut = input_tensor

    x = layers.Conv2D(filters1, (3, 3),
                      strides = strides,
                      kernel_initializer='he_normal',
                      padding = 'same')(input_tensor)
    
    x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.Activation('relu')(x)


    x = layers.Conv2D(filters2, (3,3),
                      kernel_initializer='he_normal',
                      padding = 'same')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    
    
    
    X_shortcut = layers.Conv2D(filters2, (1,1),
                      strides = strides,
                      kernel_initializer='he_normal',
                      padding = 'same')(input_tensor)
    X_shortcut = layers.BatchNormalization(axis=bn_axis)(X_shortcut)
    
    x = layers.add([x, X_shortcut])
    x = layers.Activation('relu')(x)
    return x





def ResNets(depth=20, input_shape=(32,32,3)):
    
    
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else: 
        bn_axis = 3

     
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 56 or 110 in [b])')    
    n = math.floor((depth-2)/6)
   
    img_input = layers.Input(shape=input_shape)

    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(img_input)
    x = layers.Conv2D(16, (3, 3),
                      strides=(1, 1),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)
    

    x = conv_block(x, (16,16), strides = (1,1))
    for i in range(n-1): 
        x =  identity_block(x, (16,16), (1,1))
    x = conv_block(x, (32,32), strides = (2,2))
    
    for i in range(n-1):    
        x =  identity_block(x, (32,32), (1,1))
    x = conv_block(x, (64,64), strides = (2,2))
    
    for i in range(n-1):    
        x =  identity_block(x, (64,64), (1,1))
        
    x = layers.GlobalAveragePooling2D(name='block3_pool')(x)
    x = layers.Dense(10, activation='softmax', name='fc1')(x)
   

    model = Model(img_input, x, name='resnet34')
    
    return model

