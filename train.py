import Arguments
import utils
from datapro import AbstractDataset, NILMDataloader
from model import BERT4NILM, CNNNetwork, CombinedModel
from trainer import Trainer
from utils import *
import random
from tqdm import tqdm

import argparse
import torch

para = Arguments.ukdale_params_appliance


def train(args, export_root=None, logger=None):
    # path for training data
    training_path = args.training_dataset_path
    logger.info('Training dataset: ' + training_path)

    # path for val data
    val_path = args.val_dataset_path
    logger.info('val dataset: ' + val_path)
    dataset = AbstractDataset(training_path, val_path, para)
    dataloader = NILMDataloader(args, dataset)
    train_loader, val_loader = dataloader.get_dataloaders()

    if export_root == None:
        folder_name = '-'.join('total')
        export_root = 'experiments/' + folder_name

    previous_name = 'best_acc_model'
    optimizer = None
    try:
        combined_model = torch.load(os.path.join(export_root, previous_name + '.pth'), map_location='cpu')
        # optimizer = torch.load(os.path.join(export_root, previous_name + 'optimizer' + '.pth'), map_location='cpu')
        logger.info('Successfully loaded previous model, continue training...')
    except FileNotFoundError:
        logger.info('Failed to load old model, continue training new model...')

        model = BERT4NILM(args)

        new_model1 = CNNNetwork(args)
        new_model2 = CNNNetwork(args)
        new_model3 = CNNNetwork(args)
        new_model4 = CNNNetwork(args)
        new_model5 = CNNNetwork(args)

        combined_model = CombinedModel(pretrained_model=model, new_model=new_model1,
                                       new_model2=new_model2, new_model3=new_model3, new_model4=new_model4,
                                       new_model5=new_model5)

    total_parameters = 0
    for name, param in combined_model.named_parameters():
        logger.info(f"Parameter: {name}, Requires Grad: {param.requires_grad},number of parameters{param.numel()}")
        if 'pretrained' in name:
            # if model_struct[channel] in name and 'model2' not in name and "pretrained" not in name and 'model3' not in name and 'model4' not in name and 'model5' not in name:

            total_parameters += param.numel()

    trainer = Trainer(args, combined_model, train_loader, val_loader, export_root, optimizer)
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
