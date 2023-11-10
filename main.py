import numpy as np 
import os, sys, shutil 
import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
from dataloader import DataGenerator 
from custom_trainer import Trainer 
from mann import MANN 
from utils import * 
import argparse
import datetime, random


def main(config):
    random_seed = config.random_seed

    # random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    model_name = f"mann_K{config.num_shot}_N{config.num_classes}"

    # summary writers.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs/{model_name}'
    summary_writer = tf.summary.create_file_writer(log_dir)

    data_generator = DataGenerator(num_classes=num_classes, 
                                  num_samples_per_class=num_samples + 1)

    model = MANN(num_classes=config.num_classes, 
                samples_per_class=config.num_shot + 1, 
                model_size=config.hidden_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    trainer = Trainer()
    trained_model, test_accs = trainer.train(
        model=model,
        optimizer=optimizer,
        data_generator=data_generator,
        num_classes=config.num_classes,
        num_samples=config.num_shot,
        eval_freq=eval_freq,
        meta_batch_size=config.meta_batch_size,
        summary_writer=summary_writer,
        training_steps=config.train_steps
    )

    plt.plot(range(len(test_accs)), test_accs)
    plt.xlabel("Step (x 100)")
    plt.ylabel("Test accuracy")
    plt.savefig(f"results/{model_name}.png")

    return trained_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    main(parser.parse_args())



