#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = ".."
sys.path.insert(0, prodir)


import pickle as pkl
import logging
import argparse
import random

from CMHCH.tf_models.mtl.CMHCH import CMHCH
from data_loader import DataLoader
from utility import *

CONFIG_ROOT = curdir + "/config"

seed_val = 7
np.random.seed(seed_val)
random.seed(seed_val)
tf.compat.v1.set_random_seed(seed_val)


def main():
    start_t = time.time()
    # Obtain arguments from system
    parser = argparse.ArgumentParser("Tensorflow")
    parser.add_argument(
        "--task",
        default="train",
        help="Task: Can be train or predict, the default value is train.",
    )
    parser.add_argument(
        "--data_name", default="clothes", help="Data_Name: The data you will use."
    )
    parser.add_argument(
        "--model_name", default="cmhch", help="Model_Name: The model you will use."
    )
    parser.add_argument(
        "--model_path", default="none", help="Model_Path: The model path you will load."
    )
    parser.add_argument(
        "--memory", default="0.", help="Memory: The gpu memory you will use."
    )
    parser.add_argument("--gpu", default="0", help="GPU: Which gpu you will use.")
    parser.add_argument(
        "--log_path",
        default="./logs/",
        help="path of the log file.",
    )
    parser.add_argument(
        "--ways",
        default="cmhch",
        help="decide use what kind of training method.",
    )
    parser.add_argument(
        "--use_pretrain",
        default="1",
        help="whether to use supervised method or position or deal.",
    )
    parser.add_argument(
        "--info", default="ordinary", help="information about model training."
    )

    args = parser.parse_args()

    now_time = time.strftime("%Y.%m.%d", time.localtime())

    # log directory
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # log file name setting
    log_path = (
        args.log_path
        + now_time
        + "."
        + args.info
        + "."
        + args.model_name
        + "."
        + args.data_name
        + "."
        + args.task
        + "."
        + args.ways
        + ".log"
    )

    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger("Tensorflow")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Running with args : {}".format(args))

    # get object named data_loader
    use_pre_train = 1 if args.use_pretrain == "1" else 0

    data_prepare = DataLoader(data_name=args.data_name, use_pre_train=use_pre_train)

    # Get config from file
    logger.info("Load dataset and vocab...")
    data_config_path = CONFIG_ROOT + "/data/config." + args.data_name + ".json"
    model_config_path = CONFIG_ROOT + "/model/config." + args.model_name + ".json"
    data_config = data_prepare.load_config(data_config_path)
    model_config = data_prepare.load_config(model_config_path)

    logger.info("Data config is {}".format(data_config))
    logger.info("Model config is {}".format(model_config))

    # Get config param
    model_name = model_config["model_name"]
    batch_size = model_config["batch_size"]
    epochs = model_config["total_epoch"]
    keep_prob = model_config["keep_prob"]

    is_val = model_config["is_val"]
    is_test = model_config["is_test"]
    save_best = model_config["save_best"]
    shuffle = model_config["shuffle"]

    data_name = data_config["data_name"]
    nb_classes = data_config["nb_classes"]
    task = args.task

    if shuffle == 1:
        shuffle = True
    else:
        shuffle = False

    vocab_path = curdir + "/data/" + data_name + "/vocab.pkl"

    memory = float(args.memory)
    logger.info(
        "Memory in train %s. If Memory == 0, gpu_options.allow_growth = True." % memory
    )

    # Get vocab
    with open(vocab_path, "rb") as fp:
        vocab = pkl.load(fp)

    # Get Network Framework
    if model_name == "cmhch":
        network = CMHCH(memory=memory, vocab=vocab, config_dict=model_config)
    else:
        logger.info("We can't find {}: Please check model you want.".format(model_name))
        raise ValueError(
            "We can't find {}: Please check model you want.".format(model_name)
        )

    # Set param for network
    network.set_nb_words(min(vocab.size(), data_config["nb_words"]) + 1)
    network.set_data_name(data_name)
    network.set_name(
        model_name
        + "."
        + args.info
        + ".total_epoch"
        + str(model_config["total_epoch"])
        + ".pre_epoch"
        + str(model_config["pre_epoch"])
    )
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)
    if args.task == "train":
        network.build_dir()
    network.build_graph()
    data_generator = data_prepare.data_generator
    logger.info("All values in the Network are {}".format(network.__dict__))

    if args.task == "train":
        train(
            network,
            data_generator,
            keep_prob,
            epochs,
            data_name,
            task=task,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best,
            ways=args.ways,
        )
    elif args.task == "test":
        network.test_cmhch(
            data_generator,
            data_name,
            batch_size=batch_size,
            nb_classes=nb_classes,
            test_task="test",
            model_path=args.model_path,
        )
    else:
        logger.info(
            "{}: Please check task you want, such as train or evaluate.".format(
                args.task
            )
        )
        raise ValueError(
            "{}: Please check task you want, such as train or evaluate.".format(
                args.task
            )
        )

    logger.info(
        "The whole program spends time: {}h: {}m: {}s".format(
            int((int(time.time()) - start_t) / 3600),
            int((int(time.time()) - start_t) % 3600 / 60),
            int((int(time.time()) - start_t) % 3600 % 60),
        )
    )
    print("DONE!")


def train(
    network,
    data_generator,
    keep_prob,
    epochs,
    data_name,
    task="train",
    batch_size=20,
    nb_classes=2,
    shuffle=True,
    is_val=True,
    is_test=True,
    save_best=True,
    ways="crf",
    cf_data_generator="",
):
    if ways == "mhch":
        network.train_mhch(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            task=task,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best,
        )
    elif ways == "ssa":
        network.train_ssa(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            task=task,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best,
        )
    elif ways == "cmhch":
        network.train_cmhch(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            task=task,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            is_save=True,
            save_best=save_best,
            save_frequency=10,
        )
    else:
        raise ValueError("Wrong training ways parameters: {}".format(ways))


if __name__ == "__main__":
    main()
