from label_studio_ml.model import LabelStudioMLBase

class DummyModel(LabelStudioMLBase):
    def predict(self, tasks, **kwargs):
        return []