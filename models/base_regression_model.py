class BaseRegressionModel():
    def __init__(self):
        super(BaseRegressionModel, self).__init__()

    def build_model(self, cfg):
        raise NotImplementedError
