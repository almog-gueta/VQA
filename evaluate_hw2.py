import torch
from torch.utils.data import DataLoader
from utils import main_utils, train_utils, ensemble, vision_utils
import hydra
from omegaconf import DictConfig, OmegaConf
# from utils.train_logger import TrainLogger
from dataset import MyDataset


@hydra.main(config_name="cfg")
def evaluate_hw2(cfg: DictConfig):

    # load val data
    main_utils.init(cfg)
    # logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    # logger.write(OmegaConf.to_yaml(cfg))
    main_utils.set_seed(cfg['main']['seed'])

    print('creating images file')
    vision_utils.create_vision_files(cfg)
    print('creating dataset')
    train_dataset = MyDataset(cfg, 'train', is_padding=True)
    w2idx, idx2w = train_dataset.w2idx, train_dataset.idx2w
    val_dataset = MyDataset(cfg, 'val', w2idx, idx2w, is_padding=True)
    print('creating loaders')
    # val_dataset = torch.load(cfg['main']["paths"]['val_dataset'])
    val_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                            num_workers=cfg['main']['num_workers'], collate_fn=main_utils.collate_fn)

    # load trained model-  assume that the model file is located in the script folder
    max_q_len = val_loader.dataset.max_q_length
    num_ans = val_loader.dataset.num_of_ans

    print('creating model')
    model1, model2, model3 = ensemble.load_ensemble_models(max_q_len, num_ans, cfg)

    print('evaluate & calc soft accuracy')
    soft_accuracy = ensemble.ensemble_evaluate(model1, model2, model3, val_loader)
    print(f"soft accuracy on validation set is: {soft_accuracy}")
    # return the accuracy on val set
    return soft_accuracy


evaluate_hw2()
