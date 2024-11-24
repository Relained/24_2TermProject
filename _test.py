import os
from functions import fc
from functions import db
import mariadb

base_dir = os.getcwd()
fc.create_config_json(base_dir, database={
        "name": "noDG",
        "model_name": "my_trained_model.h5",
        "user": "root",
        "password": "52455245klakla@",
        "host": "localhost",
        "port": 3306
    })
paths, database, _= fc.load_config_json(base_dir)

for _, path in paths.items():
    os.makedirs(path, exist_ok=True)
    print(f"Directory {path} created")

connection = mariadb.connect(user = database["user"],
                             password = database["password"],
                             host = database["host"],
                             port = database["port"]
                             )


print("""\nSetup Done!, put image for model input to *path*/storage/dataset (with appropriate folder tree)
Then, do make_model for making your custom layered image model!
or, you can choose your custom directory by changing config.json""")