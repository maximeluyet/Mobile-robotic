import yaml

class Config:

    @classmethod
    def load(cls, path="config.yaml"):
        ''' Load configuration from a YAML file.'''
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data
        