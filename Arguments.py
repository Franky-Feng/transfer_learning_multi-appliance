import argparse

refit_params_appliance = {
    'kettle': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 3998,
        'mean': 50,
        'std': 80, },
    'microwave': {
        'windowlength': 480,
        'on_power_threshold': 500,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,},
    'fridge': {
        'windowlength': 480,
        'on_power_threshold': 19,
        'max_on_power': 3323,
        'mean': 350,
        'std': 700,},
    'dishwasher': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 3964,
        'mean': 100,
        'std': 400,},
    'washingmachine': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 100,
        'std': 500,},
    'aggregate':
        {
            'mean': 500,
            'std': 800
        }
}

ukdale_params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 40,
        'max_on_power': 4000,
        'mean': 100,
        'std': 500,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 100,
        'max_on_power': 4000,
        'mean': 60,
        'std': 300,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 4000,
        'mean': 50,
        'std': 50,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 30,
        'max_on_power': 4000,
        'mean': 700,
        'std': 1000,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 30,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
    },
    'aggregate':
        {
            'mean': 400,
            'std': 500
        }
}

redd_params_appliance = {
    'kettle': {
        'windowlength': 480,
        'on_power_threshold': 0,
        'max_on_power': 0,
        'mean': 0,
        'std': 1,
    },
    'microwave': {
        'windowlength': 480,
        'on_power_threshold': 50,
        'max_on_power': 6000,
        'mean': 150,
        'std': 500,
    },
    'fridge': {
        'windowlength': 480,
        'on_power_threshold': 50,
        'max_on_power': 6000,
        'mean': 70,
        'std': 100,
    },
    'dishwasher': {
        'windowlength': 480,
        'on_power_threshold': 50,
        'max_on_power': 4000,
        'mean': 400,
        'std': 450,
    },
    'washingmachine': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 4000,
        'mean': 1000,
        'std': 1500,
    },
    'aggregate':
        {
            'mean': 300,
            'std': 550
        }
}

default_params_appliance = {
    'kettle': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 4000,
        'mean': 700,
        'std': 1000,
    },
    'microwave': {
        'windowlength': 480,
        'on_power_threshold': 500,
        'max_on_power': 4000,
        'mean': 500,
        'std': 800,
    },
    'fridge': {
        'windowlength': 480,
        'on_power_threshold': 19,
        'max_on_power': 4000,
        'mean': 200,
        'std': 400,
    },
    'dishwasher': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 6000,
        'mean': 700,
        'std': 1000,
    },
    'washingmachine': {
        'windowlength': 480,
        'on_power_threshold': 20,
        'max_on_power': 4000,
        'mean': 400,
        'std': 700,
    },
    'aggregate':
        {
            'mean': 522,
            'std': 814
        }
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=155)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--window_size', type=int, default=480)
parser.add_argument('--window_stride', type=int, default=120)
parser.add_argument("--hidden", type=int, default=32, help="encoder decoder hidden size")
parser.add_argument('--normalize', type=str, default='mean',
                    choices=['mean', 'minmax'])
parser.add_argument('--denom', type=int, default=2000)
parser.add_argument('--model_size', type=str, default='gru',
                    choices=['gru', 'lstm', 'dae'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--device', type=str, default='cuda',
                    choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str,
                    default='adam', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument("--logname", action="store", default='root', help="name for log")
parser.add_argument("--enable_lr_schedule", type=bool, default=False)
parser.add_argument('--training_dataset_path',type=str,default='dataset_management/refit/total/train_set.npy',
                    help='this is the directory of training dataset path')
parser.add_argument('--val_dataset_path',type=str,default='dataset_management/refit/total/val_set2.npy',
                    help='this is the directory of val dataset path')
parser.add_argument('--test_dataset_path',type=str,default='dataset_management/redd/total/test_set.npy',
                    help='this is the directory of test dataset path')
parser.add_argument('--model_path',type=str,default='model/transfer_multi_appliance_redd/best_acc_model.pth',
                    help='this is the directory of model path')
# parser.add_argument('--model_path',type=str,default='model/transfer_single_appliance_redd/m-i-c-r-o-w-a-v-e/best_acc_model.pth',
#                     help='this is the directory of model path')
parser.add_argument('--channel',type=int, default=1)
args = parser.parse_args()

appliance_name = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
