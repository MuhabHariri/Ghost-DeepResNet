import tensorflow as tf
from tensorflow.keras import layers
from src.layers import ECALayer, WeightedResidualConnection
from src.config import hidden_units, additional_layers,  Dropout_Classification, STAGES, dropout_rate



def ghost_module(
    inputs,
    num_filters: int,
    *,
    ratio: int               = 2,
    kernel_size: tuple       = (1, 1),
    dw_kernel_size: tuple    = (3, 3),
    strides: int | tuple     = 1,
):

    # ----   primary branch ----------------------------------------------
    primary_filters = num_filters // ratio
    x = layers.Conv2D(
        primary_filters, kernel_size,
        padding="same", strides=strides, use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)


    # ---- ghost branch -------------------------------------------------
    ghost = layers.DepthwiseConv2D(
        dw_kernel_size,
        padding="same",
        strides=1,
        depth_multiplier=ratio - 1,
        use_bias=False
    )(x)
    ghost = layers.BatchNormalization()(ghost)

    # ----   fuse ---------------------------------------------------------
    return layers.Concatenate()([x, ghost])



def Ghost_DeepResNet_Block(
    x,
    *,
    num_filters: int,
    kernel_size: tuple       = (1, 1),
    dw_kernel_size: tuple    = (3, 3),
    strides: tuple           = (1, 1),
    training: bool           = True,

):

    residual = x

    for _ in range(additional_layers):
        x = ghost_module(
            x,
            num_filters,
            kernel_size=kernel_size,
            dw_kernel_size=dw_kernel_size,
            strides=(1, 1)
        )
        x = tf.keras.activations.gelu(x)
        x = ECALayer(num_filters)(x)
        

    # -------- Residual alignment --------------------------------------
    if residual.shape[-1] != num_filters or strides != (1, 1):
        residual = layers.Conv2D(
            num_filters, (1, 1), strides=strides, padding="same"
        )(residual)
        residual = layers.BatchNormalization()(residual)

    x = WeightedResidualConnection()(x, residual)
    x = tf.keras.activations.gelu(x)
    return x


def Ghost_DeepResNet_Model(
    input_tensor,
    num_classes: int,
    *,
    stages_cfg: dict = STAGES,          
    dropout_rate: float = dropout_rate  
):


    x = input_tensor
    stage_names = ["stage1", "stage2", "stage3", "stage4"]

    for name in stage_names:
        params = stages_cfg[name]

        x = Ghost_DeepResNet_Block(x, **params, training=True)
        x = layers.SpatialDropout2D(dropout_rate)(x)

        if name != "stage4":            
            x = layers.MaxPooling2D(   
                pool_size=(2, 2), strides=(2, 2)
            )(x)

    # ---------------- Pool & classification head -------------------
    pooled = layers.AveragePooling2D(pool_size=(14, 14), strides=(7, 7))(x)
    pooled = layers.BatchNormalization()(pooled)
    pooled = layers.Flatten()(pooled)

    return mlp_head(pooled, hidden_units, num_classes)


def mlp_head(x, hidden_units, output_units, activation=None, Dropout_Classification=0.0):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(Dropout_Classification)(x)
    x = layers.Dense(output_units, activation='softmax')(x)
    return x


