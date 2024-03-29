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

from CEM.tf_models.mtl.CEM import CEM
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
        type=str,
        help="Task: Can be train or test, the default value is train.",
    )
    parser.add_argument(
        "--data", default="clothes", type=str, help="Data: The data you will use."
    )
    parser.add_argument(
        "--model_path", type=str, help="Model_Path: The model path you will load."
    )
    parser.add_argument(
        "--memory", default=0.0, type=float, help="Memory: The gpu memory you will use."
    )
    parser.add_argument(
        "--batch_size",
        default="64",
        type=int,
        help="Batch size: The size of each batch of data.",
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU: Which gpu you will use."
    )
    parser.add_argument(
        "--log_path",
        default="./logs/",
        type=str,
        help="path of the log file.",
    )
    parser.add_argument(
        "--model",
        default="cem",
        type=str,
        help="decide use what kind of model.",
    )
    parser.add_argument(
        "--info", default="ordinary", type=str, help="information about task."
    )
    parser.add_argument(
        "--is_only_cf",
        default=False,
        type=bool,
        help="ablation study for counterfactual.",
    )
    parser.add_argument(
        "--is_only_ssa",
        default=False,
        type=bool,
        help="ablation study for satisfaction.",
    )
    parser.add_argument(
        "--add_senti_loss",
        default=False,
        type=bool,
        help="whether add senti in loss function.",
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
        + args.model
        + "."
        + args.data
        + "."
        + args.task
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
    data_prepare = DataLoader(data=args.data)

    # Get config from file
    logger.info("Load dataset and vocab...")
    data_config_path = CONFIG_ROOT + "/data/config." + args.data + ".json"
    model_config_path = CONFIG_ROOT + "/model/config." + args.model + ".json"
    data_config = data_prepare.load_config(data_config_path)
    model_config = data_prepare.load_config(model_config_path)

    logger.info("Data config is {}".format(data_config))
    logger.info("Model config is {}".format(model_config))

    # Get config param
    batch_size = int(args.batch_size)
    epochs = model_config["total_epoch"]
    keep_prob = model_config["keep_prob"]

    is_val = model_config["is_val"]
    is_test = model_config["is_test"]
    save_best = model_config["save_best"]
    shuffle = model_config["shuffle"]
    nb_classes = data_config["nb_classes"]

    if shuffle == 1:
        shuffle = True
    else:
        shuffle = False

    vocab_path = curdir + "/data/" + args.data + "/vocab.pkl"

    memory = args.memory
    logger.info(
        "Memory in train %s. If Memory == 0, gpu_options.allow_growth = True." % memory
    )

    # Get vocab
    with open(vocab_path, "rb") as fp:
        vocab = pkl.load(fp)

    # Get Network Framework
    if args.model == "cem":
        network = CEM(
            memory=memory,
            vocab=vocab,
            config_dict=model_config,
            is_only_cf=args.is_only_cf,
            is_only_ssa=args.is_only_ssa,
            add_senti_loss=args.add_senti_loss,
            batch_size=args.batch_size,
        )
    else:
        logger.info("We can't find {}: Please check model you want.".format(args.model))
        raise ValueError(
            "We can't find {}: Please check model you want.".format(args.model)
        )

    if args.task == "test" and args.model_path == None:
        raise ValueError("Please input the model path you want to evaluate. ")

    # Set param for network
    network.set_nb_words(min(vocab.size(), data_config["nb_words"]) + 1)
    network.set_data(args.data)
    network.set_name(
        args.model
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
            data=args.data,
            task=args.task,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best,
            model=args.model,
        )
    elif args.task == "test":
        network.test_cem(
            data_generator,
            args.data,
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
    data,
    task="train",
    batch_size=20,
    nb_classes=2,
    shuffle=True,
    is_val=True,
    is_test=True,
    save_best=True,
    model="cem",
):
    if model == "cem":
        network.train_cem(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data=data,
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
        raise ValueError("Wrong training model parameters: {}".format(model))


if __name__ == "__main__":
    main()
