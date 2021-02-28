import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from models import v_models
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils import main_utils, train_utils, vision_utils
import time
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
from typing import List
import matplotlib.pyplot as plt

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, cnn_model, de_conv_model):
        super(ConvAutoencoder, self).__init__()
        self.encoder = cnn_model
        self.decoder = de_conv_model

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out


# define the deConvolution model
class DeConvolution(nn.Module):
    def __init__(self, dims):
        super(DeConvolution, self).__init__()
        ## decoder layers ##
        layers = []
        dims.reverse()
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if i % 2 == 0:
                layers.append(nn.Sequential(
                                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                                getattr(nn, "ReLU")()))
            else:
                layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, padding=1),
                    getattr(nn, "ReLU")()))

        layers.append(nn.Sequential(
            nn.Sigmoid(),
            nn.ConvTranspose2d(dims[-2], dims[-1], kernel_size=2, stride=2)))

        # if len(dims) % 2 == 0:
        #     layers.append(nn.Sequential(
        #                     nn.Sigmoid(),
        #                     nn.ConvTranspose2d(dims[-2], dims[-1], kernel_size=2, stride=2)))
        # else:
        #     layers.append(nn.Sequential(
        #         nn.Sigmoid(),
        #         nn.ConvTranspose2d(dims[-2], dims[-1], kernel_size=2)))

        self.seq = nn.Sequential(*layers)

        # self.l0 = nn.Sequential(nn.ConvTranspose2d(dims[0], dims[1], kernel_size=2, stride=2), getattr(nn, "ReLU")())
        # self.l2 = nn.Sequential(nn.ConvTranspose2d(dims[2], dims[3], kernel_size=2, stride=2), getattr(nn, "ReLU")())
        # self.l4 = nn.Sequential(nn.ConvTranspose2d(dims[4], dims[5], kernel_size=2, stride=2), getattr(nn, "ReLU")())
        # self.l6 = nn.Sequential(nn.ConvTranspose2d(dims[6], dims[7], kernel_size=2, stride=2), getattr(nn, "ReLU")())
        # self.l1 = nn.Sequential(nn.ConvTranspose2d(dims[1], dims[2], kernel_size=3, padding=1), getattr(nn, "ReLU")())
        # self.l3 = nn.Sequential(nn.ConvTranspose2d(dims[3], dims[4], kernel_size=3, padding=1), getattr(nn, "ReLU")())
        # self.l5 = nn.Sequential(nn.ConvTranspose2d(dims[5], dims[6], kernel_size=3, padding=1), getattr(nn, "ReLU")())
        # self.last = nn.Sequential(nn.Sigmoid(),nn.ConvTranspose2d(dims[-2], dims[-1], kernel_size=2, stride=2))

    def forward(self, x):
        ## decode ##
        # add transpose conv layers, with relu activation function
        # x = F.relu(self.t_conv1(x)) # [batch, 32, 28*2=56, 56]
        # # print("de_cnn1", x.shape)
        # x = F.relu(self.t_conv2(x))  # [batch, 16, 56*2=112, 112]
        # # print("de_cnn2", x.shape)
        # # output layer (with sigmoid for scaling from 0 to 1)
        # x = self.sigmoid(self.t_conv3(x)) # [batch, 3, 112*2=224, 224]
        # # print("de_cnn3", x.shape)
        x = self.seq(x)
        return x

 # not used
class convolution(nn.Module):
    def __init__(self):
        super(convolution, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [batch, 16, 224, 224]
        # print("cnn1", x.shape)
        x = self.pool(x) # # [batch, 16, 224/2=112, 112]
        # print("cnn1 pool ", x.shape)

        x = F.relu(self.conv2(x)) # [batch, 32, 112, 112]
        # print("cnn2", x.shape)
        x = self.pool(x) # [batch, 32, 112/2=56, 56]
        # print("cnn2 pool", x.shape)

        x = F.relu(self.conv3(x))  # [batch, 64, 56, 56]
        # print("cnn3", x.shape)
        x = self.pool(x)  # [batch, 64, 56/2=28, 28]
        # print("cnn3 pool", x.shape)
        return x


class CNN(nn.Module):
    def __init__(self, dims, kernel_size=3, padding=2, pool=2,
                 fc_out=1024, activation='ReLU', is_atten=False, is_autoencoder=False, resize_h=224, resize_w=224):
        super(CNN, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            # layers.append(nn.Sequential(
            #                 nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding), #, stride=2
            #                 nn.BatchNorm2d(out_dim),
            #                 getattr(nn, activation)(),
            #                 nn.MaxPool2d(pool)))
            if i % 2 == 0 or i == (len(dims) - 2):
                layers.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),  # , stride=2
                    nn.BatchNorm2d(out_dim),
                    getattr(nn, activation)(),
                    nn.MaxPool2d(pool)))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),  # , stride=2
                    nn.BatchNorm2d(out_dim),
                    getattr(nn, activation)()))

        self.seq = nn.Sequential(*layers)

        if len(dims) % 2 == 0:
            denominator = 2**(len(dims) / 2)
        else:
            denominator = 2 ** (int(len(dims) / 2)+1)
        fc_in = int(((resize_h / denominator)) * ((resize_w / denominator)))  if is_atten else int(((resize_h / denominator)) * ((resize_w / denominator)) * dims[-1])
        self.fc_out = fc_in if is_atten else fc_out
        self.is_atten = is_atten
        # if not self.is_atten:
        #     self.fc = nn.Linear(fc_in, self.fc_out)
        self.v_out_dim = dims[-1] if is_atten else fc_out
        self.is_autoencoder = is_autoencoder

    def forward(self, img: Tensor):
        # img: [batch_size, 3, resize_h, resize_w]

        out = self.seq(img) # [batch_size, conv_out_l2, resize_h / denominator, resize_w / denominator]

        if self.is_autoencoder:
            return out




class autoencoder_dataset(Dataset):
    def __init__(self, cfg, logger=None) -> None:
        super(autoencoder_dataset, self).__init__()
        # Set variables
        self.data_name = "train"
        self.dataset_path = "./data/autoencoder_dataset.pth" # "/home/student/hw2/autoencoder_dataset.pth"

        # preprocess vision
        print('we are at preprocess vision')
        self.imgs_file_path = cfg['vision_utils'][f'{self.data_name}_file_path']
        self.img_id2idx = self.create_img_id2idx()

        # Create list of entries
        print('we are at creating entries')
        self.entries = self._get_entries()

    def __getitem__(self, index: int) -> torch:
        v_idx = self.img_id2idx[self.entries[index]]
        imgs_file = h5py.File(self.imgs_file_path, mode='r')
        v = torch.from_numpy(imgs_file.get('imgs')[v_idx, :, :, :].astype('float32')) # [3, resize_h, resize_w]
        return v

    def __len__(self) -> int:
        return len(self.entries)


    def create_img_id2idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.imgs_file_path, mode='r') as imgs_file:
            img_ids = imgs_file['img_ids'][()]
        img_id2idx = {id: i for i, id in enumerate(img_ids)}
        return img_id2idx

    def _save(self):
        torch.save(self, self.dataset_path)
        print(f"saved dataset in: {self.dataset_path}")

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []
        for v_id in self.img_id2idx.keys():
            entries.append(v_id)

        return entries


def init_v_model(cfg):
    resizes_dict = {'resize_h': cfg['dataset']['resize_h'], 'resize_w': cfg['dataset']['resize_w']}
    cfg_dict = {
        "dims": [3, 32, 32, 64, 64, 128, 128, 256, 256], #, 512, 512], #[3, 16, 32, 64, 128 ,256, 512, 1024], # [3, 32, 64, 128, 256],
        "kernel_size": 3,
        "padding": 1,
        "pool": 2,
        "fc_out": 1024,
        "activation": 'ReLU',
        "is_atten": False,
        "is_autoencoder": True,
    }
    v_model_params = dict(cfg_dict, **resizes_dict)

    v_model = CNN(**v_model_params)
    return v_model, cfg_dict['dims']


def create_loaders(cfg):
    # train_dataset = torch.load("/home/student/hw2/autoencoder_dataset.pth") #82,000 samples
    train_dataset = torch.load("./data/autoencoder_dataset.pth")  # 82,000 samples
    # sampler = torch.utils.data.RandomSampler(data_source=train_dataset, num_samples=)
    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    return train_loader

def plot_loss(loss_list):
    plt.plot(list(loss_list[:10]), c="red", label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()



@hydra.main(config_name="cfg")
def main(cfg: DictConfig) -> None:

    # create dataset- run only one time
    train_dataset = autoencoder_dataset(cfg)
    train_dataset._save()

    # initialize the NN
    print("initialize the NN")
    encoder_model, dims = init_v_model(cfg=cfg) # convolution()
    decoder_model = DeConvolution(dims)
    model = ConvAutoencoder(encoder_model, decoder_model)
    if torch.cuda.is_available():
        model = model.cuda()

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 150

    # create loaders
    print("create loaders")
    train_loader = create_loaders(cfg)

    # metrics
    cumulative_loss = []

    # create dir to save model weights every 5 epochs
    main_utils.make_dir('/home/student/hw2/autoencoder_8_layers/')

    print("starting training")
    for epoch in tqdm(range(1, n_epochs + 1)):
        start_epoch = time.time()
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for i, v in enumerate(train_loader):
            if torch.cuda.is_available():
                v = v.cuda()  # [batch_size, 3, resize_h, resize_w]
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(v)
            # calculate the loss
            loss = criterion(outputs, v)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * v.size(0)

            if i % 1000 == 0:
                print(f"done {i} batches in {(time.time()-start_epoch) / 60} mins")

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        cumulative_loss.append(train_loss)
        print('Epoch: {} took {} mins \tTraining Loss: {:.6f}'.format(
            epoch,
            (time.time()-start_epoch) / 60,
            train_loss
        ))

        if epoch % 5 == 0:
            # save trained CNN model
            model_dict = model.encoder.state_dict()
            torch.save(model_dict, f"/home/student/hw2/autoencoder_8_layers/trained_cnn_{epoch}.pth")
        # if epoch % 10 == 0:
        #     plot_loss(cumulative_loss)


if __name__ == '__main__':
    main()