from torch import nn, Tensor


class CNN(nn.Module):
    def __init__(self, dims, kernel_size=3, padding=2, pool=2,
                 fc_out=1024, activation='ReLU', is_atten=False, is_autoencoder=False, resize_h=224, resize_w=224):
        super(CNN, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            if len(dims) <= 5:
                layers.append(nn.Sequential(
                                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding), #, stride=2
                                nn.BatchNorm2d(out_dim),
                                getattr(nn, activation)(),
                                nn.MaxPool2d(pool)))
            else:
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
        if len(dims) <= 5:
            denominator = 2 ** (len(dims)-1)
        fc_in = int(((resize_h / denominator)) * ((resize_w / denominator)))  if is_atten else int(((resize_h / denominator)) * ((resize_w / denominator)) * dims[-1])
        self.fc_out = fc_in if is_atten else fc_out
        self.is_atten = is_atten
        if not self.is_atten:
            self.fc = nn.Linear(fc_in, self.fc_out)
        self.v_out_dim = dims[-1] if is_atten else fc_out
        self.is_autoencoder = is_autoencoder

    def forward(self, img: Tensor):
        # img: [batch_size, 3, resize_h, resize_w]

        out = self.seq(img) # [batch_size, conv_out_l2, resize_h / denominator, resize_w / denominator]

        if self.is_autoencoder:
            return out 

        if self.is_atten:
            out = out.view(out.shape[0], out.shape[1], -1) # [batch_size, conv_out_dim, resize_h/denominator*resize_w/denominator]
            return out
        else:
            out = out.view(out.shape[0], -1) # [batch_size, conv_out_dim * resize_h / denominator * resize_w / denominator]
            out = self.fc(out) # [batch_size, 1024]
            return out
