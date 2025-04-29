import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam

class Solver:
    """Solver for training GAN models"""
    
    def __init__(self, generator, discriminator, learning_rate=0.0002, beta1=0.5, lambda_l1=100.0, l2_reg=1e-5):
        """
        Initialize the GAN solver
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            learning_rate: Learning rate for Adam optimizer
            beta1: Beta1 parameter for Adam optimizer
            lambda_l1: Weight for L1 loss
            l2_reg: L2 regularization strength
        """
        self.generator = generator
        self.discriminator = discriminator
        
        # Define optimizers
        self.gen_optimizer = Adam(learning_rate=learning_rate, beta_1=beta1)
        self.disc_optimizer = Adam(learning_rate=learning_rate, beta_1=beta1)
        
        # Define loss functions - Updated based on research papers for MRI-to-CT conversion
        self.adv_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.l1_loss_fn = tf.keras.losses.MeanAbsoluteError()
        
        # Loss weights - L1 loss has higher weight to preserve structure
        self.lambda_l1 = lambda_l1  # High weight on L1 for preserving anatomical details
        self.perceptual_weight = 5.0  # Moderate weight on perceptual loss
        
        # Regularization strength
        self.l2_reg = l2_reg
        
        # Initialize metrics
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
        self.l1_loss_metric = tf.keras.metrics.Mean(name='l1_loss')
        self.adv_loss_metric = tf.keras.metrics.Mean(name='adv_loss')
        self.perceptual_loss_metric = tf.keras.metrics.Mean(name='perceptual_loss')
    
    def _apply_l2_reg(self, model):
        """Apply L2 regularization to model weights"""
        reg_loss = 0
        for weight in model.trainable_weights:
            if 'kernel' in weight.name:  # Only apply to kernel weights, not biases
                reg_loss += tf.nn.l2_loss(weight)
        return self.l2_reg * reg_loss
    
    @tf.function
    def _train_generator_step(self, mr_images, real_ct):
        """Single training step for generator"""
        # Generate fake CT images
        with tf.GradientTape() as tape:
            fake_ct = self.generator(mr_images, training=True)
            
            # Discriminator prediction on fake CT
            fake_output = self.discriminator(fake_ct, training=False)
            
            # Adversarial loss (Generator wants discriminator to think fakes are real)
            adv_loss = self.adv_loss_fn(tf.ones_like(fake_output), fake_output)
            
            # L1 loss (pixel-wise difference between real and fake)
            l1_loss = self.l1_loss_fn(real_ct, fake_ct)
            
            # Compute gradient differences for perceptual-like loss
            gen_gradients = tf.image.image_gradients(fake_ct)
            target_gradients = tf.image.image_gradients(real_ct)
            grad_diff_x = tf.reduce_mean(tf.abs(gen_gradients[0] - target_gradients[0]))
            grad_diff_y = tf.reduce_mean(tf.abs(gen_gradients[1] - target_gradients[1]))
            perceptual_loss = grad_diff_x + grad_diff_y
            
            # Total generator loss
            gen_loss = 0.5 * adv_loss + self.lambda_l1 * l1_loss + self.perceptual_weight * perceptual_loss
            
            # Add L2 regularization
            gen_loss += self._apply_l2_reg(self.generator)
            
        # Calculate gradients and update generator
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        # Update metrics
        self.gen_loss_metric.update_state(gen_loss)
        self.adv_loss_metric.update_state(adv_loss)
        self.l1_loss_metric.update_state(l1_loss)
        self.perceptual_loss_metric.update_state(perceptual_loss)
        
        return gen_loss, adv_loss, l1_loss, perceptual_loss
    
    @tf.function
    def _train_discriminator_step(self, mr_images, real_ct):
        """Single training step for discriminator"""
        with tf.GradientTape() as tape:
            # Generate fake CT images
            fake_ct = self.generator(mr_images, training=False)
            
            # Discriminator prediction on real CT
            real_output = self.discriminator(real_ct, training=True)
            # Discriminator prediction on fake CT
            fake_output = self.discriminator(fake_ct, training=True)
            
            # Calculate discriminator loss
            real_loss = self.adv_loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.adv_loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
            # Add L2 regularization
            disc_loss += self._apply_l2_reg(self.discriminator)
            
        # Calculate gradients and update discriminator
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        # Update metrics
        self.disc_loss_metric.update_state(disc_loss)
        
        return disc_loss
    
    def train_generator(self, mr_images, ct_images):
        """Train generator on a batch of data"""
        # Convert inputs to tensors if they're not already
        mr_images = tf.convert_to_tensor(mr_images, dtype=tf.float32)
        ct_images = tf.convert_to_tensor(ct_images, dtype=tf.float32)
        
        # Train generator
        gen_loss, adv_loss, l1_loss, perceptual_loss = self._train_generator_step(mr_images, ct_images)
        
        return gen_loss.numpy(), adv_loss.numpy(), l1_loss.numpy(), perceptual_loss.numpy()
    
    def train_discriminator(self, mr_images, ct_images):
        """Train discriminator on a batch of data"""
        # Convert inputs to tensors if they're not already
        mr_images = tf.convert_to_tensor(mr_images, dtype=tf.float32)
        ct_images = tf.convert_to_tensor(ct_images, dtype=tf.float32)
        
        # Train discriminator
        disc_loss = self._train_discriminator_step(mr_images, ct_images)
        
        return disc_loss.numpy()
    
    def validate_generator(self, mr_images, ct_images):
        """Validate generator on a batch of data"""
        # Convert inputs to tensors if they're not already
        mr_images = tf.convert_to_tensor(mr_images, dtype=tf.float32)
        ct_images = tf.convert_to_tensor(ct_images, dtype=tf.float32)
        
        # Generate fake CT images
        fake_ct = self.generator(mr_images, training=False)
        
        # Discriminator prediction on fake CT
        fake_output = self.discriminator(fake_ct, training=False)
        
        # Adversarial loss
        adv_loss = self.adv_loss_fn(tf.ones_like(fake_output), fake_output)
        
        # L1 loss 
        l1_loss = self.l1_loss_fn(ct_images, fake_ct)
        
        # Perceptual loss
        gen_gradients = tf.image.image_gradients(fake_ct)
        target_gradients = tf.image.image_gradients(ct_images)
        grad_diff_x = tf.reduce_mean(tf.abs(gen_gradients[0] - target_gradients[0]))
        grad_diff_y = tf.reduce_mean(tf.abs(gen_gradients[1] - target_gradients[1]))
        perceptual_loss = grad_diff_x + grad_diff_y
        
        # Total generator loss
        gen_loss = 0.5 * adv_loss + self.lambda_l1 * l1_loss + self.perceptual_weight * perceptual_loss
        
        return gen_loss.numpy(), adv_loss.numpy(), l1_loss.numpy(), perceptual_loss.numpy()
    
    def validate_discriminator(self, mr_images, ct_images):
        """Validate discriminator on a batch of data"""
        # Convert inputs to tensors if they're not already
        mr_images = tf.convert_to_tensor(mr_images, dtype=tf.float32)
        ct_images = tf.convert_to_tensor(ct_images, dtype=tf.float32)
        
        # Generate fake CT images
        fake_ct = self.generator(mr_images, training=False)
        
        # Discriminator prediction on real CT
        real_output = self.discriminator(ct_images, training=False)
        
        # Discriminator prediction on fake CT
        fake_output = self.discriminator(fake_ct, training=False)
        
        # Calculate discriminator loss
        real_loss = self.adv_loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.adv_loss_fn(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss
        
        return disc_loss.numpy()
    
    def reset_metrics(self):
        """Reset metrics between epochs"""
        self.gen_loss_metric.reset_states()
        self.disc_loss_metric.reset_states()
        self.l1_loss_metric.reset_states()
        self.adv_loss_metric.reset_states()
        self.perceptual_loss_metric.reset_states()
    
    # Optional method to update learning rate during training
    def update_learning_rate(self, new_lr):
        """Update learning rate for both optimizers"""
        self.gen_optimizer.lr.assign(new_lr)
        self.disc_optimizer.lr.assign(new_lr) 