class TrainerConfig:
    # optimization parameters
    n_out_channels = 12
    n_hidden = 14
    n_layer = 2
    kernel_size = 3

    def __init__(self, time_steps, n_out_channels, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
