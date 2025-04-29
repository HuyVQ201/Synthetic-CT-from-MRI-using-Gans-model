import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def Generator(input_shape=(256, 256, 1), l2_reg=0.0001, name='generator'):
    """U-Net based generator for MRI to CT conversion with residual connections for better gradient flow"""
    
    # Set up regularizer
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    # Input
    inputs = Input(input_shape)
    
    # Encoder
    # Block 1
    conv1 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizer)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizer)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    # Block 6
    up6 = Conv2D(512, (2, 2), padding='same', kernel_regularizer=regularizer)(UpSampling2D(size=(2, 2))(drop5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    # Add residual connection
    conv6 = Conv2D(512, (1, 1), padding='same', kernel_regularizer=regularizer)(conv6)  # Match dimensions
    conv6 = tf.keras.layers.add([conv6, drop4])  # Add residual connection
    
    # Block 7
    up7 = Conv2D(256, (2, 2), padding='same', kernel_regularizer=regularizer)(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    # Add residual connection
    conv7 = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizer)(conv7)  # Match dimensions
    conv7 = tf.keras.layers.add([conv7, conv3])  # Add residual connection
    
    # Block 8
    up8 = Conv2D(128, (2, 2), padding='same', kernel_regularizer=regularizer)(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    # Add residual connection
    conv8 = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizer)(conv8)  # Match dimensions
    conv8 = tf.keras.layers.add([conv8, conv2])  # Add residual connection
    
    # Block 9
    up9 = Conv2D(64, (2, 2), padding='same', kernel_regularizer=regularizer)(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer)(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    # Add residual connection
    conv9 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizer)(conv9)  # Match dimensions
    conv9 = tf.keras.layers.add([conv9, conv1])  # Add residual connection
    
    # Output - Changed from tanh to sigmoid for better CT intensity range
    pre_output = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizer)(conv9)
    pre_output = BatchNormalization()(pre_output)
    pre_output = Activation('relu')(pre_output)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizer)(pre_output)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model

def Discriminator(input_shape=(256, 256, 2), l2_reg=0.0001, name='discriminator'):
    """PatchGAN discriminator for GAN training"""
    
    # Set up regularizer
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    # Input
    inputs = Input(input_shape)
    
    # Block 1
    conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(inputs)
    conv1 = Activation('leaky_relu')(conv1)
    
    # Block 2
    conv2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('leaky_relu')(conv2)
    
    # Block 3
    conv3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('leaky_relu')(conv3)
    
    # Block 4
    conv4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('leaky_relu')(conv4)
    
    # Block 5
    conv5 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('leaky_relu')(conv5)
    
    # Output
    outputs = Conv2D(1, (4, 4), padding='same', activation='sigmoid', kernel_regularizer=regularizer)(conv5)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model 