import Arguments
import utils
from datapro import AbstractDataset, NILMDataloader, AbstractDataset_appliance
from model import BERT4NILM, CNNNetwork, CombinedModel, shared_layer, CombinedModel2
from trainer import Trainer, Trainer_one_appliance
from utils import *
import random
from tqdm import tqdm
import argparse
import torch

# dataset params
para = Arguments.redd_params_appliance


def train(args, export_root=None, logger=None):
    # which appliance to train
    channel = args.channel
    # path for training data
    training_path = args.training_dataset_path
    logger.info('Training dataset: ' + training_path)

    # path for val data
    val_path = args.val_dataset_path
    logger.info('val dataset: ' + val_path)
    dataset = AbstractDataset_appliance(training_path, val_path, para, channel)
    dataloader = NILMDataloader(args, dataset)
    train_loader, val_loader = dataloader.get_dataloaders()

    if export_root == None:
        folder_name = '-'.join(appliance_name[channel])
        export_root = 'single_appliance_redd/' + folder_name

    previous_name = 'best_acc_model'

    optimizer = None
    try:
        combined_model = torch.load(os.path.join(export_root, previous_name + '.pth'), map_location='cpu')
        # optimizer = torch.load(os.path.join(export_root, previous_name + 'optimizer' + '.pth'), map_location='cpu')
        logger.info('Successfully loaded previous model, continue training...')
    except FileNotFoundError:
        logger.info('Failed to load old model, continue training new model...')
        model = shared_layer(args)
        new_model1 = CNNNetwork(args)
        combined_model = CombinedModel2(pretrained_model=model, new_model1=new_model1)

    for name, param in combined_model.named_parameters():
        logger.info(f"Parameter: {name}, Requires Grad: {param.requires_grad},number of parameters{param.numel()}")

    trainer = Trainer_one_appliance(args, combined_model, train_loader, val_loader, export_root, optimizer)
    trainer.train(logger)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

torch.set_default_tensor_type(torch.DoubleTensor)
args = Arguments.args

if __name__ == "__main__":
    fix_random_seed_as(args.seed)
    # get_user_input(args)
    utils.mkdir("log/" + 'total')
    logger = utils.setup_log('total', args.logname)
    train(args, logger=logger)
