import ml_collections


def get_HFTrans5_16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (320, 160, 80)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans5_32_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (320, 160, 80)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans4_16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (256, 128, 64)
    config.n_dims = 3
    config.n_skip = 3
    return config

    
def get_HFTrans4_32_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (32, 64, 128, 256)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (128, 64, 32)
    config.skip_channels = (512, 256, 128,)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans_16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (16, 32, 64, 128)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (64, 32, 16)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans_16b2s_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (64, 128, 256, 512)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (64, 32, 16)
    config.skip_channels = (64, 32, 16)
    config.n_dims = 3
    config.n_skip = 3
    return config

def get_HFTrans_64_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (2, 2, 2)
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 8
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_size = 1

    config.conv_first_channel = 128
    config.encoder_channels = (64, 128, 256, 512)
    config.down_factor = 2
    config.down_num = 3
    config.decoder_channels = (256, 128, 64)
    config.skip_channels = (256, 128, 64)
    config.n_dims = 3
    config.n_skip = 3
    return config