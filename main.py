import torch
import random
import argparse
import numpy as np
import train
from configure import get_default_hyperparams


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')

# Tasks
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei',
                    help='dataset to use (default: mosei)')
parser.add_argument('--feature', type=str, default='glove',
                    help='text feature type: glove or Bert')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')


args = parser.parse_args()



if __name__ == '__main__':
    setup_seed(args.seed)
    print(f'seed: {args.seed}')
    dataset = str.lower(args.dataset.strip())
    hyperparams  = get_default_hyperparams(args.dataset, args.aligned, args.feature)
    train.train_and_eval(hyperparams)
