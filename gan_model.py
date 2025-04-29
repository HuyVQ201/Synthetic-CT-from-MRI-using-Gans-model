import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class GANModel:
    def __init__(self, args, input_shape=(256, 256, 1)):
        self.args = args
        self.input_shape = input_shape
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.combined = self._build_combined()
        
        # Optimizers
        self.d_optimizer = Adam(learning_rate=0.00005, beta_1=0.5)
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Compile models
        self.discriminator.compile(optimizer=self.d_optimizer, loss='binary_crossentropy')
        self.generator.compile(optimizer=self.g_optimizer, loss='binary_crossentropy')
        self.combined.compile(optimizer=self.g_optimizer, loss='binary_crossentropy')

    def _build_generator(self):
        """Build Generator network"""
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        conv1 = Conv2D(64, 3, padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        
        conv2 = Conv2D(128, 3, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Decoder
        up1 = UpSampling2D(size=(2, 2))(pool4)
        up1 = Concatenate(axis=3)([up1, conv4])
        conv5 = Conv2D(512, 3, padding='same')(up1)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        
        up2 = UpSampling2D(size=(2, 2))(conv5)
        up2 = Concatenate(axis=3)([up2, conv3])
        conv6 = Conv2D(256, 3, padding='same')(up2)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        
        up3 = UpSampling2D(size=(2, 2))(conv6)
        up3 = Concatenate(axis=3)([up3, conv2])
        conv7 = Conv2D(128, 3, padding='same')(up3)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        
        outputs = Conv2D(1, 1, activation='tanh')(conv7)
        
        return Model(inputs=inputs, outputs=outputs)

    def _build_discriminator(self):
        """Build Discriminator network"""
        inputs = Input(shape=self.input_shape)
        
        conv1 = Conv2D(64, 3, padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        
        conv2 = Conv2D(128, 3, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        flat = Flatten()(pool4)
        dense1 = Dense(1024)(flat)
        dense1 = BatchNormalization()(dense1)
        dense1 = Activation('relu')(dense1)
        
        outputs = Dense(1, activation='sigmoid')(dense1)
        
        return Model(inputs=inputs, outputs=outputs)

    def _build_combined(self):
        """Build Combined GAN network"""
        self.discriminator.trainable = False
        inputs = Input(shape=self.input_shape)
        generated = self.generator(inputs)
        validity = self.discriminator(generated)
        return Model(inputs=inputs, outputs=[generated, validity])

    def train(self, x_train, y_train, batch_size=32, epochs=100):
        """Train the GAN"""
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training history
        history = {
            'gen_total_loss': [],
            'gen_adv_loss': [],
            'gen_l1_loss': [],
            'disc_loss': []
        }
        
        # Lambda coefficient for L1 loss (high weight for detail preservation)
        lambda_l1 = 100.0
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            real_imgs = y_train[idx]
            
            # Generate fake images
            gen_imgs = self.generator.predict(imgs)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            # Instead of just the basic binary cross-entropy, calculate and use both
            # adversarial loss and L1 loss combined for better image quality
            g_loss_adv = self.combined.train_on_batch(imgs, [real_imgs, valid])
            
            # Manually calculate L1 loss
            g_loss_l1 = np.mean(np.abs(real_imgs - gen_imgs))
            
            # Calculate perceptual loss (simplified by calculating gradient differences)
            # This helps preserve structural details of the brain, especially bone boundaries
            gen_gradients_x = np.gradient(gen_imgs.squeeze(), axis=1)
            gen_gradients_y = np.gradient(gen_imgs.squeeze(), axis=0)
            real_gradients_x = np.gradient(real_imgs.squeeze(), axis=1)
            real_gradients_y = np.gradient(real_imgs.squeeze(), axis=0)
            
            grad_diff_x = np.mean(np.abs(gen_gradients_x - real_gradients_x))
            grad_diff_y = np.mean(np.abs(gen_gradients_y - real_gradients_y))
            g_loss_perceptual = grad_diff_x + grad_diff_y
            
            # Total generator loss
            g_loss = g_loss_adv + lambda_l1 * g_loss_l1 + 10.0 * g_loss_perceptual
            
            # Update history
            history['gen_total_loss'].append(g_loss)
            history['gen_adv_loss'].append(g_loss_adv)
            history['gen_l1_loss'].append(g_loss_l1)
            history['disc_loss'].append(d_loss)
            
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G adv: {g_loss_adv:.4f}] [G L1: {g_loss_l1:.4f}] [G perceptual: {g_loss_perceptual:.4f}] [G total: {g_loss:.4f}]")
            
            # Save generated images periodically
            if epoch % 10 == 0:
                self.save_generated_images(epoch, imgs, gen_imgs)
                
        return history

    def save_generated_images(self, epoch, input_imgs, generated_imgs):
        """Save generated images"""
        import matplotlib.pyplot as plt
        import os
        
        # Create directory if it doesn't exist
        os.makedirs('generated_images', exist_ok=True)
        
        # Plot and save images
        plt.figure(figsize=(10, 5))
        
        # Plot input images
        plt.subplot(1, 2, 1)
        plt.imshow(input_imgs[0].reshape(self.input_shape[0], self.input_shape[1]), cmap='gray')
        plt.title('Input MRI')
        
        # Plot generated images
        plt.subplot(1, 2, 2)
        plt.imshow(generated_imgs[0].reshape(self.input_shape[0], self.input_shape[1]), cmap='gray')
        plt.title('Generated CT')
        
        plt.savefig(f'generated_images/epoch_{epoch}.png')
        plt.close() 