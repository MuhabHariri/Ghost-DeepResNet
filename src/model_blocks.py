from src.layers import ECALayer, WeightedResidualConnection
from src.config import hidden_units, additional_layers,  Dropout_Classification, STAGES, dropout_rate
import torch
import torch.nn as nn
import numpy as np




class GhostModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int = 2,
        kernel_size: int | tuple[int, int] = 1,
        dw_kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        activation_fn: nn.Module | None = None,
    ):
        super().__init__()

        if ratio < 2:
            raise ValueError("ratio must be >= 2")

        self.activation_fn = activation_fn  # can be None

        # ----primary branch -------------------------------------------------
        primary_filters = out_channels // ratio           
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                primary_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size),
                bias=True,
            ),
            nn.BatchNorm2d(primary_filters, eps=1e-3, momentum=0.01), 
        )

        # ---cheap (ghost) branch -------------------------------------------
        ghost_channels = primary_filters * (ratio - 1)      # depth_multiplier = ratio-1
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(
                primary_filters,
                ghost_channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=dw_kernel_size // 2 if isinstance(dw_kernel_size, int) else tuple(k // 2 for k in dw_kernel_size),
                groups=primary_filters,                     # depth-wise
                bias=True,
            ),
            nn.BatchNorm2d(ghost_channels, eps=1e-3, momentum=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # primary features
        primary = self.primary_conv(x)
        if self.activation_fn is not None:
            primary = self.activation_fn(primary)

        # cheap ghost features
        ghost = self.ghost_conv(primary)

        # concatenate along the channel dimension
        return torch.cat([primary, ghost], dim=1)


class GhostDeepResNetBlock(nn.Module):
    """
    PyTorch replica of the Keras `GhostDeepResNetBlock`.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        dw_kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        additional_layers: int = 1,
        dropout_rate: float = 0.05,        
        ratio: int = 2                     
    ):
        super().__init__()

        self.activation = nn.GELU()

        # -------- build the sequence of (Ghost ➜ GELU ➜ ECA) layers ------------
        layer_defs = []

        # first pair 
        layer_defs.append((GhostModule(in_channels,
                                       out_channels,
                                       ratio=ratio,
                                       kernel_size=kernel_size,
                                       dw_kernel_size=dw_kernel_size,
                                       stride=1,
                                       activation_fn=None),   # GELU is applied outside
                           ECALayer(out_channels)))

        # middle pairs
        for _ in range(additional_layers):
            layer_defs.append((GhostModule(out_channels,
                                           out_channels,
                                           ratio=ratio,
                                           kernel_size=kernel_size,
                                           dw_kernel_size=dw_kernel_size,
                                           stride=1,
                                           activation_fn=None),
                               ECALayer(out_channels)))

        # final pair
        layer_defs.append((GhostModule(out_channels,
                                       out_channels,
                                       ratio=ratio,
                                       kernel_size=kernel_size,
                                       dw_kernel_size=dw_kernel_size,
                                       stride=1,
                                       activation_fn=None),
                           ECALayer(out_channels)))

        self.ghost_eca_layers = nn.ModuleList(
            [nn.ModuleList([ghost, eca]) for ghost, eca in layer_defs]
        )

        # ----residual adaptation path ------------------------------
        if in_channels != out_channels or stride != 1:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=True),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.residual_conv = nn.Identity()

        # learnable scalar skip-connection
        self.wskip = WeightedResidualConnection()

    # --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)

        # main path:  Ghost → GELU → ECA  (× N)
        for ghost, eca in self.ghost_eca_layers:
            x = ghost(x)
            x = self.activation(x)
            x = eca(x)

        # weighted skip add  + final GELU
        x = self.wskip(x, residual)
        x = self.activation(x)
        return x


class MLPHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_units: list[int],
        out_features: int,
        activation=nn.ReLU,         
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev = in_features
        for h in hidden_units:
            layers.append(nn.Linear(prev, h, bias=True))  
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, out_features, bias=True))  

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class Ghost_DeepResNet_Model(nn.Module):

    def __init__(
        self,
        num_classes: int,
        hidden_units: list[int],
        dropout_rate: float = 0.05,
        ratio: int = 2,                    
        stages_config: dict = None,       
    ):
        super().__init__()
        
        if stages_config is None:
            stages_config = STAGES

        # ---------- stage 1 --------------------------------------------------
        stage1_config = stages_config["stage1"]
        self.block1 = GhostDeepResNetBlock(
            3, stage1_config["num_filters"],
            kernel_size=stage1_config["kernel_size"],
            dw_kernel_size=stage1_config["dw_kernel_size"],
            stride=stage1_config["strides"][0] if isinstance(stage1_config["strides"], tuple) else stage1_config["strides"],
            additional_layers=1,
            dropout_rate=dropout_rate,
            ratio=ratio
        )
        self.drop1  = nn.Dropout2d(dropout_rate)           # SpatialDropout2D
        self.pool1  = nn.MaxPool2d(2, 2)

        # ---------- stage 2 --------------------------------------------------
        stage2_config = stages_config["stage2"]
        self.block2 = GhostDeepResNetBlock(
            stage1_config["num_filters"], stage2_config["num_filters"],
            kernel_size=stage2_config["kernel_size"],
            dw_kernel_size=stage2_config["dw_kernel_size"],
            stride=stage2_config["strides"][0] if isinstance(stage2_config["strides"], tuple) else stage2_config["strides"],
            additional_layers=1,
            dropout_rate=dropout_rate,
            ratio=ratio
        )
        self.drop2  = nn.Dropout2d(dropout_rate)
        self.pool2  = nn.MaxPool2d(2, 2)

        # ---------- stage 3 --------------------------------------------------
        stage3_config = stages_config["stage3"]
        self.block3 = GhostDeepResNetBlock(
            stage2_config["num_filters"], stage3_config["num_filters"],
            kernel_size=stage3_config["kernel_size"],
            dw_kernel_size=stage3_config["dw_kernel_size"],
            stride=stage3_config["strides"][0] if isinstance(stage3_config["strides"], tuple) else stage3_config["strides"],
            additional_layers=1,
            dropout_rate=dropout_rate,
            ratio=ratio
        )
        self.drop3  = nn.Dropout2d(dropout_rate)
        self.pool3  = nn.MaxPool2d(2, 2)

        # ---------- stage 4 --------------------------------------------------
        stage4_config = stages_config["stage4"]
        self.block4 = GhostDeepResNetBlock(
            stage3_config["num_filters"], stage4_config["num_filters"],
            kernel_size=stage4_config["kernel_size"],
            dw_kernel_size=stage4_config["dw_kernel_size"],
            stride=stage4_config["strides"][0] if isinstance(stage4_config["strides"], tuple) else stage4_config["strides"],
            additional_layers=1,
            dropout_rate=dropout_rate,
            ratio=ratio
        )
        self.drop4  = nn.Dropout2d(dropout_rate)           

        # ---------- head -----------------------------------------------------
        self.final_pool = nn.AvgPool2d(kernel_size=14, stride=7)
        self.bn         = nn.BatchNorm2d(stage4_config["num_filters"], eps=1e-3, momentum=0.01)  
        self.head       = MLPHead(
            in_features=stage4_config["num_filters"] * 3 * 3,       
            hidden_units=hidden_units,
            out_features=num_classes,
            dropout_rate=0.3
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x); x = self.drop1(x); x = self.pool1(x)
        x = self.block2(x); x = self.drop2(x); x = self.pool2(x)
        x = self.block3(x); x = self.drop3(x); x = self.pool3(x)
        x = self.block4(x); x = self.drop4(x)

        x = self.final_pool(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        return self.head(x)
