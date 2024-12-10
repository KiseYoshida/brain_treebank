from models import register_model
import torch
from models.base_regression_model import BaseRegressionModel

@register_model("classification_regression_model")
class ClassificationRegressionModel(BaseRegressionModel):
    def __init__(self):
        super(ClassificationRegressionModel, self).__init__()

    def build_model(self, cfg):
        self.cfg = cfg
        #TODO
