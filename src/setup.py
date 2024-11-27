import os
from src.utils import fc

# from functions import db

base_dir = os.getcwd()
fc.create_config_json(base_dir, database={
        "name": "noDG",
        "model_name": "my_trained_model.pth",
        "user": "root",
        "password": "<PASSWORD>", # change to your password
        "host": "localhost",
        "port": 3306
    })
paths, database, _= fc.load_config_json(base_dir)

for _, path in paths.items():
    os.makedirs(path, exist_ok=True)
    print(f"Directory {path} created")

# connection = db.connect_mariadb(database["user"], database["password"], database["host"], database["port"])
# db.create_db(connection)
# db.create_tables(connection)
# connection.close()

print("""\nSetup Done!, put image for model input to *path*/storage/dataset (with appropriate folder tree)
Then, do make_model for making your custom layered image model!
or, you can choose your custom directory by changing config.json""")

# 초기 config 생성 및 db 생성 작업, storage 생성