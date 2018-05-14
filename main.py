import os
import sys
import numpy as np
import random as rn

# GAN Variants
from GAN import GAN
from CGAN import CGAN
from WGAN_GP import WGAN_GP
from BEGAN import BEGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse
import configparser

sys.path.insert(0, '/home/DADOS1/gabriel/random/slack-bot/')

from slackbot import SlackBot


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    """parsing and configuration"""

    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'BEGAN', 'WGAN_GP'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='increase output verbosity')

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""

    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, \
        'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, \
        'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, \
        'dimension of noise vector must be larger than or equal to one'

    return args


def main():
    """main"""

    # Slack Bot
    config = configparser.ConfigParser()
    config.read('slack.config')
    bot = SlackBot(token=config['SLACK']['token'],
                   channel_name=config['SLACK']['channel_name'])

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    models = [GAN, CGAN, WGAN_GP, BEGAN]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir,
                            bot=bot,
                            verbosity=args.verbosity)
        if gan is None:
            bot.send_message(text="<!channel> ERROR!\n\n"
                             "[!] There is no option for " + args.gan_type)
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        bot.send_message(text="[*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        bot.send_message(text="[*] Testing finished!")


if __name__ == '__main__':
    main()
