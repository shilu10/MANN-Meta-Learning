import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
import datetime
from tqdm import tqdm
from data_generator import DataGenerator


class Trainer:
    """
        This class used to train the model (tensorflow model), it 
        uses the custom training loop for training the tensorflow 
        model.
        methods:
            _train_step(type: private)
            _test_step(type: private)
            train(type: public)
    """
    def __init__(self):
        self.test_accs = []

    @tf.function
    def _train_step(images: tf.Tensor, 
                  labels: tf.Tensor, 
                  model: tf.keras.Model, 
                  optim: tf.keras.Optimizer, 
                  eval: bool =False):

        """
            this method used to calculate the loss, and uses the 
            optimizer to update the model parameters
            Params:
                images(type: tf.Tensor): it is input(training input(independent variables)) of our model
                labels(type: tf.Tensor): it is response varaible(training outpur) of our model
                optim(type: tf.keras.Optimizer): optimizer, used to update the model params.
                model(type: tf.keras.Model): model, that we need to train,

            Returns(type: Tuple)
                this method returns the prediction of our model and loss over the training data
                prediction
        """

        with tf.GradientTape() as tape:
            predictions = model(images, labels)
            loss = model.loss_function(predictions, labels)
        if not eval:
            gradients = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(gradients, model.trainable_variables))
        return predictions, loss

    @tf.function 
    def _test_step(self, 
                images: tf.Tensor, 
                labels: tf.Tensor, 
                model: tf.keras.Model):
        """
            this method used to calculate the test loss, and prediction
            of the testing data.
            Params:
                images(type: tf.Tensor): it is input(testing input(independent variables)) of our model
                labels(type: tf.Tensor): it is response varaible(testing outpur) of our model
                model(type: tf.keras.Model): model, that we need to train,

            Returns(type: Tuple)
                this method returns the prediction of our model and loss over the testing data
                prediction
        """

        predictions = model(images, labels)
        loss = model.loss_function(predictions, labels)
        
        return predictions, loss 

    def train(model: tf.keras.Model, 
              optimizer: tf.keras.Optimizer, 
              data_generator: DataGenerator, 
              num_classes: int, 
              num_samples: int, 
              model_name: str, 
              summary_writer, 
              training_steps):

        """
            this method, combines the train_step and test_step, and implements the training step 
            it has a logis for implementing epochs.
            Params:
                model(type: tf.keras.Model): model, that we need to train,
                optimizer(type: tf.keras.Optimizer): optimizer, used to update the model params.
                data_generator(type: DataGenerator): custom DataGenerator, that is used for sampling
                                                     each epochs.
                num_classes(dtype: int): Number of classes in a task in support set and query set.
                num_samples(dtype: int): Number of samples/ instance per classes in a task. 
                model_name(dtype: str): Name of the model variant.
                summary_writer: It is used for recording the metrics for each epochs/ steps
                training_step(dtype: int): Total number of epochs/ steps

            Returns(type: Tuple)
                this method returns the trained model and test_accs calculated over epochs.

        """

        for step in tqdm(range(training_steps), total=training_steps):
            # train step
            images, labels = data_generator.sample_batch('train', meta_train_batch_size)
            train_prediction, train_loss = self._train_step(images, labels, model, optim)

            if (step + 1) % config.get("log_every") == 0:
                print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)

                # testing
                images, labels = data_generator.sample_batch('test', meta_test_batch_size)
                test_pred, test_loss = self._test_step(images, labels, model)

                print("Train Loss:", train_loss.numpy(), "Test Loss:", test_loss.numpy())

                test_pred = tf.reshape(test_pred, [-1, num_samples + 1, num_classes, num_classes])
                test_pred = tf.math.argmax(test_pred[:, -1, :, :], axis=2)
                test_label = tf.math.argmax(labels[:, -1, :, :], axis=2)

                test_acc = tf.reduce_mean(tf.cast(tf.math.equal(test_pred, test_label), tf.float32)).numpy()
                print("Test Accuracy", test_acc)

                self.test_accs.append(test_acc)

                with summary_writer.as_default():
                    tf.summary.scalar('Train Loss', train_loss.numpy(), step)
                    tf.summary.scalar('Test Loss', test_loss.numpy(), step)
                    tf.summary.scalar('Meta-Test Accuracy', test_acc, step)


        return model, self.test_accs