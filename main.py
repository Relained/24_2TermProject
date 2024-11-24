import os
import functions.db as db
import functions.fc as fc

base_dir = os.getcwd()
paths, database, allowed_ext = fc.load_config_json(base_dir)
# connection = db.connect_mariadb(database["user"], database["password"], database["host"], database["port"])
