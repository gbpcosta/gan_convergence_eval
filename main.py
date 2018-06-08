import os
import sys
import numpy as np
import random as rn
import argparse
import configparser

os.environ["PYTHONHASHSEED"] = '42'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '/home/DADOS1/gabriel/random/slack-bot/')

# GAN Variants
from DCGAN import DCGAN
# from CGAN import CGAN
from WGAN_GP import WGAN_GP
from BEGAN import BEGAN
from EBGAN import EBGAN

from utils.utils import show_all_variables
from utils.utils import check_folder

import tensorflow as tf

from slackbot import SlackBot


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


def parse_args():
    """parsing and configuration"""

    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='DCGAN',
                        choices=['DCGAN', 'BEGAN', 'WGAN_GP', 'EBGAN'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'celeba'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of noise vector')
    parser.add_argument('--compute_metrics_it', type=int, default=300,
                        help='At which iterations the evaluation'
                             'metrics should be computed.')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='_outputs/checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='_outputs/results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='_outputs/logs',
                        help='Directory name to save training logs')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID')
    parser.add_argument('--slack', action='store_true',
                        help='Activate Slack bot!')
    parser.add_argument('--redo', action='store_true',
                        help='Redo training from the start regardless of'
                             'finding checkpoint')
    parser.add_argument('-v', '--verbosity',
                        action='count', default=0,
                        help='Increase output verbosity')

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

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # Slack Bot
    if args.slack:
        config = configparser.ConfigParser()
        config.read('slack.config')
        bot = SlackBot(token=config['SLACK']['token'],
                       channel_name=config['SLACK']['channel_name'])
    else:
        bot = None

    gpu_options = tf.GPUOptions(visible_device_list=args.gpu_id,
                                allow_growth=True)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
        gpu_options=gpu_options,
        allow_soft_placement=True)

    # open session
    models = [DCGAN, WGAN_GP, BEGAN, EBGAN]
    with tf.Session(config=session_conf) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            compute_metrics_it=args.compute_metrics_it,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir,
                            gpu_id=args.gpu_id,
                            bot=bot,
                            redo=args.redo,
                            verbosity=args.verbosity)
        if gan is None:
            print("<!channel> ERROR!\n\n"
                  "[!] There is no option for " + args.gan_type)
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print("[*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print("[*] Testing finished!")


if __name__ == '__main__':
    main()
