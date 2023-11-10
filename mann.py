import tensorflow as tf 
from tensorflow.keras import layers 
import numpy as np 


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class, model_size=128):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(model_size, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####

        #print(tf.shape(input_images).as_list())
       # print()
        B, K, N, DIMS = input_images.get_shape().as_list()
        K = K - 1

        input_images = tf.cast(input_images, dtype=tf.float64)
        input_labels = tf.cast(input_labels, dtype=tf.float64)

        # support set and query set
        support_set_images = input_images[:, :K]
        query_set_images = input_images[:, -1]
        query_set_images = tf.reshape(query_set_images, (-1, 1, N, DIMS))

        support_set_labels = input_labels[:, :K]

        # for true label
        query_set_labels = input_labels[:, -1]

        concat_support_set_images = tf.concat([support_set_images, support_set_labels], axis=-1)
        zero_matrix = tf.zeros((B, 1, N, N), dtype=tf.float64)
        concat_query_set_images = tf.concat([query_set_images, zero_matrix], axis=-1)

        # reshaping K, N -> K*N for sequential modelling
        # concat_support_set_images = tf.reshape(concat_support_set_images, (-1, K*N, N + DIMS))
        # concat_query_set_images = tf.reshape(concat_query_set_images, (-1, 1*N, N + DIMS))

        concat_images = tf.concat([concat_support_set_images, concat_query_set_images], axis=1)
        K = K+1
        concat_images = tf.reshape(concat_images, (-1, N*K, N+DIMS))

        x = self.layer1(concat_images)
        out = self.layer2(x)

        out = tf.reshape(out, (-1, K, N, N))

        #############################
        return out

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss|
        """
        #############################
        #### YOUR CODE GOES HERE ####
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels[:, -1, :, :], logits=preds[:, -1, :, :]))
        #############################


if __name__ == '__main__':
    input_images = tf.zeros((10, 3, 5, 768))
    input_labels = tf.zeros((10, 3, 5, 5)) 
    num_classes = 5 
    num_samples_per_class = 3 
    model_size = 128

    mann_model = MANN(num_classes, num_samples_per_class, model_size)
    model_output = mann_model(input_images, input_labels)

    print(model_output.shape)
