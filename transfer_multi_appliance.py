import Arguments
import utils
from datapro import AbstractDataset, NILMDataloader
from model import BERT4NILM, CNNNetwork, CombinedModel
from trainer import Trainer, Trainer_single
from utils import *
import random
import torch
channel = 0
appliance = appliance_name[channel]
para = Arguments.ukdale_params_appliance
model_struct = ['model', 'model2', 'model3', 'model4', 'model5', ]


def train(args, export_root=None, logger=None):
    # path for training data
    training_path = 'dataset_management/ukdale/' + 'total' + '/train_set2.npy'
    logger.info('Training dataset: ' + training_path)

    # path for val data
    val_path = 'dataset_management/ukdale/' + 'total' + '/val_set2.npy'
    logger.info('val dataset: ' + val_path)
    dataset = AbstractDataset(training_path, val_path, para)
    dataloader = NILMDataloader(args, dataset)
    train_loader, val_loader = dataloader.get_dataloaders()

    if export_root == None:
        folder_name = '-'.join(appliance)
        export_root = 'experiments/' + folder_name

    try:
        combined_model = torch.load(os.path.join(export_root, 'best_acc_model0.pth'), map_location='cpu')
        logger.info('Successfully loaded previous model, continue training...')
    except FileNotFoundError:
        logger.info('Failed to load old model, training new model...')
        # folder_name = '-'.join('dishwasher')
        # export_root2 = 'experiments/' + folder_name
        export_root2 = args.model_path
        combined_model = torch.load(os.path.join(export_root2, 'best_acc_model.pth'), map_location='cpu')

        for name, param in combined_model.named_parameters():
            # if model_struct[channel] in name and 'model2' not in name and "pretrained" not in name and 'model3' not in name and 'model4' not in name and 'model5' not in name:
            if model_struct[channel] in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in combined_model.named_parameters():
            logger.info(f"Parameter: {name}, Requires Grad: {param.requires_grad},number of parameters{param.numel()}")

    trainer = Trainer_single(args, combined_model, train_loader, val_loader, export_root, channel)
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

