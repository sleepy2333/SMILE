import os

class HyperParams(object):
    def __init__(self):
        #task
        self.dataset = None
        self.aligned = None
        self.feature = None

        #training
        self.batch_size = None
        self.lr         = None
        self.when       = None
        self.scheduler  = None
        self.max_epochs = None
        self.early_stop = None
        self.model_path = None
        self.threshold = None

    def __str__(self):
        return 'dataset: {}\n aligned: {}\n feature: {}\n batch_size: {}\n lr: {}\n when: {}\n max_epochs: {}\n early_stop: {}\n model_path: {}'.format(
                self.dataset, self.aligned, self.feature, self.batch_size, self.lr, self.when, self.max_epochs, self.early_stop, self.model_path
        )


def get_default_hyperparams(dataset, aligned, feature):
    config = HyperParams()
    config.dataset = dataset
    config.aligned = aligned
    config.feature = feature

    if dataset == 'mosei':
        config.batch_size = 256
        config.lr         = 1e-3
        config.when       = 20
        config.scheduler  = True
        config.max_epochs = 50
        config.model_path = os.path.join('./models', dataset, 'aligned' if aligned else 'unaligned', feature+'.pt')

    else:
        config.batch_size = 128
        config.lr         = 2e-3
        config.when       = 20
        config.scheduler  = True
        config.max_epochs = 100
        config.model_path = os.path.join('./models', dataset, 'aligned' if aligned else 'unaligned', feature+'.pt')
        


    config.early_stop = 8

    return config