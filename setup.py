import os
from src import utils

# 초기 config 생성 및 불러오기
utils.create_config_json()
path, database_setting = utils.load_config_json()

# 폴더 생성
utils.setup_storage(path)

# DB 생성

# connection = db.connect_mariadb(database["user"], database["password"], database["host"], database["port"])
# db.create_db(connection)
# db.create_tables(connection)
# connection.close()

print("""\nSetup Done!, put image for model input to *path*/storage/dataset (with appropriate folder tree)
Then, do make_model for making your custom layered image model!
or, you can choose your custom directory by editting config.json""")

# 초기 config 생성 및 db 생성 작업, storage 생성