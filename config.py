from pathlib import Path

class Config():
    # PARAMETERS HAD TO BE UPPERCASE !

    root = Path(__file__).resolve().parents[0]

    RMS = 0.3


    INTERVAL_SAVE_MODEL = 20000
    INTERVAL_UPDATE_LR = 1_000
    INTERVAL_TENSORBOARD_PLOT = 1000
    INTERVAL_TENSORBOARD = 100
    LOAD_PATH = None
    LOAD_IDX = 0
    SIGNAL_LENGTH = 250
    DEVICE = 'cuda'
    BATCH_SIZE = 4
    LR = 1e-3
    LR_INIT = 1e-3
    LR_DECAY = 0.5
    LR_STEP = 1_000_000
    LR_DELAY = 5_000


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
