"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'paths': {
            'train_loader': str,
            'val_loader': str,
            'train_dataset': str,
            'val_dataset': str,
            'logs': str,
        },
        "model_names": {
            "q_model_name": str,
            "v_model_name": str,
            "vqa_model_name": str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        # 'dropout': float,
        # 'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': float,
        },
    },
    'main_utils': {
        'qa_path': str,
        'task': str,
        'dataset': str,
    },
    "vision_utils": {
      "train_file_path": str,
      "val_file_path": str,
      "num_train_imgs": int,
      "num_val_imgs": int,
    },
    'dataset': {
        'max_q_length': int,
        'resize_h': int,
        'resize_w': int,
        'resize_int': int,
        'filter_ans_threshold': int,
    },
    "q_model": {
        'lstm': {
            "vocab_size": int,
            "emb_dim": int,
            "hidden_dim": int,
            "num_layer": int,
            "num_hid": int,
            "output_dim": int,
            "activation": str,
            "dropout": float,
            "is_atten": bool,
        },
        "attention_lstm": {
            "vocab_size": int,
            "emb_dim": int,
            "hidden_dim": int,
            "num_layer": int,
            "num_hid": int,
            "output_dim": int,
            "activation": str,
            "dropout": float,
            "is_atten": bool,
        },
    },
    "v_model": {
        "cnn": {
            "dims": list,
            "kernel_size": int,
            "padding": int,
            "pool": int,
            "fc_out": int,
            "activation": str,
            "is_atten": bool,
            "is_autoencoder": bool,
        },
        "attention_cnn": {
            "dims": list,
            "kernel_size": int,
            "padding": int,
            "pool": int,
            "fc_out": int,
            "activation": str,
            "is_atten": bool,
            "is_autoencoder": bool,
        },
    },
    "atten_model": {
      "projected_dim": int,
    },
    "vqa_model": {
      "basic_lstm_cnn": {
        "activation": str,
        "num_hid": int,
        "dropout": float,
        "is_concat": bool,
        },
      "atten_lstm_cnn": {
        "activation": str,
        "num_hid": int,
        "dropout": float,
        "is_concat": bool,
        },
    },
}