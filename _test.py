import os
import src.model as m
import src.utils as utils

path , _ = utils.load_config_json()
unclassified_dir = os.path.join(path, "storage/unclassified")

#m.make_model()

device_model = m.load_model()

m.classify_from_folder(device_model, unclassified_dir)