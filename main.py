"""
Main file
We will run the whole program from here
"""
import time

import torch
from torch.utils.data import DataLoader
import hydra
import _pickle as cPickle
from torch.autograd import Variable

from train import train
from dataset import MyDataset
from utils import main_utils, train_utils, vision_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

@hydra.main(config_name="cfg")
def main(cfg: DictConfig, preprocess_data=True, create_images_h5_file=True):
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    #------Only run 1 time in order to create the h5 file-----:
    # create h5 file with the images, separate files for train, val
    if create_images_h5_file:
        logger.write('--------creating vision files--------')
        start = time.time()
        vision_utils.create_vision_files(cfg)
        logger.write(f'time of creating images files: {time.time()-start}')

    if preprocess_data:
        logger.write('--------preprocess data--------')
        # Load dataset
        train_dataset = MyDataset(cfg, 'train', is_padding=True)

        w2idx, idx2w = train_dataset.w2idx, train_dataset.idx2w
        val_dataset = MyDataset(cfg, 'val', w2idx, idx2w, is_padding=True)

        # save a cPickle
        # with open(cfg['main']["paths"]['train_dataset'], 'wb') as f:
        #     cPickle.dump(train_dataset, f)
        # with open(cfg['main']["paths"]['val_dataset'], 'wb') as f:
        #     cPickle.dump(val_dataset, f)

        # save as torch pth
        train_dataset._save()
        val_dataset._save()

    else:
        logger.write("--------loading datasets--------")
        # load as cPickle
        # train_dataset = cPickle.load(open(cfg['main']["paths"]['train_dataset'], 'rb'))
        # val_dataset = cPickle.load(open(cfg['main']["paths"]['val_dataset'], 'rb'))
        # load as torch pth
        train_dataset = torch.load(cfg['main']["paths"]['train_dataset'])
        val_dataset = torch.load(cfg['main']["paths"]['val_dataset'])

    logger.write('--------create data loaders--------')
    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'], collate_fn=main_utils.collate_fn)
    val_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                            num_workers=cfg['main']['num_workers'], collate_fn=main_utils.collate_fn)

    # logger.write(f'len of train loader: {len(train_loader) * cfg["train"]["batch_size"]}, '
    #       f'len of val loader: {len(val_loader) * cfg["train"]["batch_size"]}')
    # 2127 val samples dont have answers, train num samples: 443760, val num samples: 214368

    # Init model
    logger.write(f'--------init model---------')
    max_q_len = train_loader.dataset.max_q_length
    num_ans = train_loader.dataset.num_of_ans
    q_name, v_name, vqa_name = cfg['main']['model_names'].values()

    for model_name in ['no_pretrain', 'pretrain_4_layers', 'pretrain_8_layers']:

        model = main_utils.init_models(q_name, v_name, vqa_name, cfg, max_q_len, num_ans, model_name).model

        # Add gpus_to_use in cfg- not relevant, we have 1 GPU
        if cfg['main']['parallel']:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model = model.cuda()

        logger.write(main_utils.get_model_string(model))

        # Run model
        logger.write(f'--------train model---------')
        train_params = train_utils.get_train_params(cfg)


        # Report metrics and hyper parameters to tensorboard
        metrics = train(model, train_loader, val_loader, train_params, logger, model_name)
        hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

        logger.report_metrics_hyper_params(hyper_parameters, metrics)
    



if __name__ == '__main__':

    main()

