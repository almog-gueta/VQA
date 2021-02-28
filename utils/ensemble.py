import torch
from utils import main_utils, train_utils
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def load_ensemble_models(max_q_len, num_ans, cfg):
    """
    Init and load weights of all ensemble models
    :return: tuple of all ensemble models
    """

    # PRETRAINED 8 CNN LAYERS
    # load trained model-  assume that the model file is located in the script folder
    q_name, v_name, vqa_name, = 'attention_lstm', 'attention_cnn', 'atten_lstm_cnn'
    cfg1 = cfg.copy()
    cfg1["v_model"]["attention_cnn"]["dims"] = [3, 32, 32, 64, 64, 128, 128, 256, 256]
    model1 = main_utils.init_models(q_name, v_name, vqa_name, cfg1, max_q_len, num_ans).model
    # model1.load_state_dict(torch.load("/home/student/hw2/logs/saved_models/6_layers_ready/model_dict_epoch_10.pth", map_location=lambda storage, loc: storage))
    model1.load_state_dict(
        torch.load("./saved_models/pretrained_8_layers_saved_model.pth", map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        model1 = model1.cuda()
    model1.eval()

    # PRETRAINED 4 CNN LAYERS
    # load trained model-  assume that the model file is located in the script folder
    q_name, v_name, vqa_name, = 'attention_lstm', 'attention_cnn', 'atten_lstm_cnn'
    cfg2 = cfg.copy()
    cfg2["v_model"]["attention_cnn"]["dims"] = [3, 32, 64, 128, 256]
    model2 = main_utils.init_models(q_name, v_name, vqa_name, cfg2, max_q_len, num_ans).model
    # model2.load_state_dict(torch.load("/home/student/hw2/logs/saved_models/full_run_47/model_dict_epoch_12.pth", map_location=lambda storage, loc: storage))
    model2.load_state_dict(
        torch.load("./saved_models/pretrained_4_layers_saved_model.pth", map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        model2 = model2.cuda()
    model2.eval()

    # NO PRETRAIN, 8 CNN LAYERS
    # load trained model-  assume that the model file is located in the script folder
    q_name, v_name, vqa_name, = 'attention_lstm', 'attention_cnn', 'atten_lstm_cnn'
    cfg3 = cfg.copy()
    cfg3["v_model"]["attention_cnn"]["dims"] = [3, 32, 32, 64, 64, 128, 128, 256, 256]
    model3 = main_utils.init_models(q_name, v_name, vqa_name, cfg3, max_q_len, num_ans).model
    # model3.load_state_dict(torch.load("/home/student/hw2/logs/saved_models/no_pretrain_8_layers/model_dict_epoch_13.pth",
    #                                   map_location=lambda storage, loc: storage))
    model3.load_state_dict(
        torch.load("./saved_models/no_pretrain_saved_model.pth", map_location=lambda storage, loc: storage))
    if torch.cuda.is_available():
        model3 = model3.cuda()
    model3.eval()

    return model1, model2, model3




def ensemble_evaluate(model1, model2, model3, val_loader):
    soft_accuracy = 0
    with torch.no_grad():
        for i, (v, q, a) in enumerate(val_loader):
            if torch.cuda.is_available():
                v = v.cuda()  # [batch_size, 3, resize_h, resize_w]
                q = (q[0].cuda(), q[1])  # questions: [batch_size, 19], q_lens: [batch_size, 1]
                a = a.cuda()  # [batch_size, num_of_ans]

            y_hat1 = model1((v, q))
            y_hat2 = model2((v, q))
            y_hat3 = model3((v, q))

            avg_score = ((y_hat1.add(y_hat2)).add(y_hat3.mul(0.6))).mul(1.0 / 2.6)

            soft_accuracy += train_utils.compute_soft_accuracy_with_logits(avg_score, a).sum().item()

    soft_accuracy /= len(val_loader.dataset)
    soft_accuracy *= 100
    return soft_accuracy


