import torch.nn as nn


class MLPPProjector(nn.Module):
    def __init__(
        self,
        ft_dim,
        hidden_dim=4096,
        bottleneck_dim=512,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ft_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class SimCLRProjector(nn.Module):
    def __init__(
        self,
        ft_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ft_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class TReXProjector(nn.Module):
    def __init__(
        self,
        ft_dim,
        input_l2_norm=True,
        hidden_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        **kwargs,
    ):
        super().__init__()

        # list of MLP layers
        layers = []

        if input_l2_norm:
            layers.append(L2Norm(dim=1))

        # hidden layers
        _in_dim = ft_dim
        for _ in range(hidden_layers):
            layers.append(MLPLayer(_in_dim, hidden_dim))
            _in_dim = hidden_dim

        # bottleneck layer
        layers.append(nn.Linear(_in_dim, bottleneck_dim, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return nn.functional.normalize(x, dim=self.dim, p=2)



PROJECTORS = {
    "trex": TReXProjector,
    "simclr": SimCLRProjector,
    "mlpp": MLPPProjector,
}
