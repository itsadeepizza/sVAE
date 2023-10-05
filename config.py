from pathlib import Path

class Config():
    # PARAMETERS HAD TO BE UPPERCASE !

    root = Path(__file__).resolve().parents[0]

    DROPOUT_RATE  = 0.5
    SIGNAL_LENGTH = 250
    FREQ_LENGTH   = 257
    LATENT_DIM    = 64
    # LAYER_SIZES = [ [SIGNAL_LENGTH, FREQ_LENGTH],
    #                 [50 , 50 ],
    #                 []



    def __init__(self):

        self.set_derivate_parameters()

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.PATH_DATASET = str(config.root / 'dataset')
        config.ROOT_RUNS    = str(config.root)


    def get(self, key, default_return_value=None):
        """Safe metod to get an attribute. If the attribute does not exist it returns
        None or a specified default value"""
        if hasattr(self, key):
            return self.__getattribute__(key)
        else:
            return default_return_value

selected_config = Config()

if __name__ == '__main__':
    pass
