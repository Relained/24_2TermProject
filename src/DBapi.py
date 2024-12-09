import os
import mariadb
from contextlib import contextmanager
from src.utils import load_config_json
from src.model import CustomDataset

paths, database, _ = load_config_json(os.getcwd())
Dataset = CustomDataset(os.path.join(paths, "storage/dataset"))
database_name = database["name"]

# DB 연결
def connect_mariadb(user, password, host, port):
    try:
        connection = mariadb.connect(
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Successfully connected to MariaDB")
        return connection
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return None

# 컨텍스트 매니저를 이용하여 cursor 객체를 안전하게 관리
@contextmanager
def get_cursor(connection):
    """
    컨텍스트 매니저를 이용하여 cursor 객체를 안전하게 관리.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(f"USE {database_name};")
        yield cursor
    except mariadb.Error as e:
        print(f"Database error: {e}")
        raise e
    finally:
        cursor.close()

# 트랜잭션을 관리하는 컨텍스트 매니저
@contextmanager
def transaction(connection):
    """
    트랜잭션을 관리하는 컨텍스트 매니저.
    """
    try:
        yield
        connection.commit()
    except Exception:
        connection.rollback()
        raise

# 레코드 존재 확인 메소드
def record_exists(cursor, table, conditions):
    """
    주어진 조건을 기반으로 레코드가 존재하는지 확인.
    Args:
        cursor: MariaDB cursor 객체
        table: 테이블 이름 (예: "class")
        conditions: 조건 딕셔너리 (예: {"name": "Test", "p_id": 1})
    Returns:
        bool: 존재 여부
    """
    where_clause = " AND ".join([f"{key} = %s" for key in conditions.keys()])
    values = tuple(conditions.values())
    query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause};"
    cursor.execute(query, values)
    return cursor.fetchone()[0] > 0


# 매핑 테이블을 트리 구조로 변환
def mapping_tree(mtable):
    tree = {}

    for class_id, path in mtable.items():
        parts = path.split("/")
        current_level = tree

        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

    return tree

# DB 생성
def create_db(connection):
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
        connection.commit()
        print(f"Successfully created database {database_name}")
    except mariadb.Error as e:
        connection.rollback()
        print(f"Database error: {e}")
    finally:
        cursor.close()

# Table 생성
def create_tables(connection):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            # class 테이블 생성
            # 재귀형 구조
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS class (
                id INT AUTO_INCREMENT PRIMARY KEY,
                p_id INT,
                name VARCHAR(255) NOT NULL,
                description VARCHAR(255),
                FOREIGN KEY (p_id) REFERENCES class(id)
                );"""
            )
            print(f"Created table 'class'.")

            # tag 테이블 생성
            # 재귀형 구조
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tag (
                id INT AUTO_INCREMENT PRIMARY KEY,
                p_id INT,
                name VARCHAR(255) NOT NULL,
                FOREIGN KEY (p_id) REFERENCES tag(id)
                );"""
            )
            print(f"Created table 'tag'.")

            # image 테이블 생성
            # learned value로 사용중인 모델의 이미지 학습 여부 체크
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image (
                id INT AUTO_INCREMENT PRIMARY KEY,
                class_id INT,
                name VARCHAR(255),
                file_path VARCHAR(255) NOT NULL,
                learned BOOLEAN NOT NULL
                );"""
            )
            print(f"Created table 'image'.")

            # image - tag 다대다 테이블 생성
            # image 및 tag 삭제시 연관 데이터 삭제
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_tag (
                image_id INT NOT NULL,
                tag_id INT NOT NULL,
                PRIMARY KEY(image_id, tag_id),
                FOREIGN KEY (image_id) REFERENCES image(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
                );"""
            )
            print(f"Created table 'image_tags'.")

            # annotation 일대다 테이블 생성
            # image 삭제시 연관 데이터 삭제
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_id INT NOT NULL,
                x1 FLOAT NOT NULL,
                y1 FLOAT NOT NULL,
                x2 FLOAT NOT NULL,
                y2 FLOAT NOT NULL,
                label VARCHAR(255) NOT NULL,
                FOREIGN KEY (image_id) REFERENCES image(id) ON DELETE CASCADE
                );"""
            )
            print(f"Created table 'annotation'.")

        print("All tables successfully created.")
    except mariadb.Error as e:
        print(f"Error Occurred: {e}")

# CustomDataset 객체를 바탕으로 class 테이블에 클래스 구조를 insert
def insert_class_from_dataset(connection, dataset=Dataset):
    """
    train_dataset을 기반으로 데이터베이스에 클래스 구조를 삽입합니다.

    Args:
        connection: 데이터베이스 연결 객체.
        dataset: CustomDataset 객체, 클래스 이름과 ID 정보를 포함.
    """
    try:
        # 데이터베이스 트랜잭션 및 커서 사용
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            # class_to_name 매핑을 사용하여 클래스 삽입
            for class_id, class_name in dataset.class_to_name.items():
                # 부모 ID는 없으므로 기본 None
                p_id = None

                # 중복 확인
                if record_exists(cursor, "class", {"name": class_name, "p_id": p_id}):
                    print(f"Skipped (already exists): {class_name} (Parent ID: {p_id})")
                else:
                    # 중복되지 않은 경우 삽입
                    cursor.execute("""
                        INSERT INTO class (name, p_id)
                        VALUES (%s, %s);
                    """, (class_name, p_id))
                    print(f"Inserted: {class_name} (Parent ID: {p_id})")

        print("Class structure successfully inserted.")
        return True
    except Exception as e:
        print(f"Error inserting class structure: {e}")
        return False


# class 객체 read
def read_class(connection, class_id=None, p_id=None):
    try:
        with get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if class_id:
                cursor.execute("""
                SELECT * FROM class WHERE id = %s;
                """, (class_id,))
            elif p_id:
                cursor.execute("""
                SELECT * FROM class WHERE p_id = %s;
                """, (p_id,))
            else:
                cursor.execute("""
                SELECT * FROM class;
                """)

            rows = cursor.fetchall()

            if not rows:
                print("No class found.")
                return []

            print(f"retrieved {len(rows)} rows.")
            return rows

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return []

# class 객체 update
def update_class(connection, class_id, name=None, p_id=None, description=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            updates = []
            values = []

            if name:
                updates.append("name = %s")
                values.append(name)

            if p_id is not None: # 부모 ID가 명시적으로 None인 경우
                updates.append("p_id = %s")
                values.append(p_id)

            if description:
                updates.append("description = %s")
                values.append(description)

            if not updates:
                print("No updates.")
                return False

            query = f"UPDATE class SET {', '.join(updates)} WHERE id = %s;"
            values.append(class_id)
            cursor.execute(query, tuple(values))

        print(f"Class with ID {class_id} successfully updated.")
        return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# class 객체 delete
def delete_class(connection, class_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            cursor.execute(f"USE {database_name};")
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "class", {"id": class_id}):
                print(f"Class '{class_id}' not found.")
                return False

            cursor.execute("DELETE FROM class WHERE id = %s;", (class_id,))
        print(f"Class with ID {class_id} successfully deleted.")
        return True
    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False


# tag 객체 insert
def insert_tag(connection, name, p_id=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")
            conditions = {"name": name}
            if p_id is not None:
                conditions["p_id"] = p_id

            if record_exists(cursor, "tag", conditions):
                print(f"Tag '{name}' already exists.")
                return None

            if p_id is not None:
                cursor.execute("""
                INSERT INTO tag (name, p_id)
                VALUES (%s, %s);
                """, (name, p_id))
            else:
                cursor.execute("""
                INSERT INTO tag (name)
                VALUES (%s);
                """, (name,))
            tag_id = cursor.lastrowid
            print(f"Tag with ID {tag_id} successfully inserted.")
            return tag_id

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return None

# tag 객체 read
def read_tag(connection, tag_id=None, p_id=None):
    try:
        with get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if tag_id is not None:
                query = f"SELECT name FROM tag WHERE id = %s;"
                cursor.execute(query, (tag_id,))
            elif p_id is not None:
                query = f"SELECT name FROM tag WHERE p_id = %s;"
                cursor.execute(query, (p_id,))
            else:
                query = f"SELECT name FROM tag;"
                cursor.execute(query)

            rows = cursor.fetchall()

            if not rows:
                print("No tag found.")
                return []

            print(f"retrieved {len(rows)} rows.")
            return [rows[0] for row in rows]

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return []

# tag 객체 update
def update_tag(connection, tag_id, name=None, p_id=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            updates = []
            values = []

            if name:
                updates.append("name = %s")
                values.append(name)

            if p_id is not None:
                if not record_exists(cursor, "tag", {"id": p_id}):
                    print(f"Parent tag with ID {p_id} does not exist.")
                    return False
                updates.append("p_id = %s")
                values.append(p_id)

            if not updates:
                print("No updates.")
                return False

            query = f"UPDATE tag SET {', '.join(updates)} WHERE id = %s;"
            values.append(tag_id)
            cursor.execute(query, tuple(values))

            if cursor.rowcount == 0:
                print(f"No tag found with ID {tag_id}.")
                return False

        print(f"Tag with ID {tag_id} successfully updated.")
        return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# tag 객체 delete
def delete_tag(connection, tag_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "tag", {"id": tag_id}):
                print(f"Tag '{tag_id}' not found.")
                return False

            cursor.execute("DELETE FROM tag WHERE id = %s;", (tag_id,))
            if cursor.rowcount == 0:
                print(f"Tag with ID {tag_id} could not be deleted.")
                return False

        print(f"Tag with ID {tag_id} successfully deleted.")
        return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False


# image 객체 insert
def insert_image(connection, class_id, file_path, learned=False, name=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")
            if record_exists(cursor, "image", {"class_id": class_id, "file_path": file_path}):
                print(f"Image '{file_path}' already exists.")
                return None

            cursor.execute("""
            INSERT INTO image (class_id, file_path, learned, name)
            VALUES (%s, %s, %s, %s);
            """, (class_id, file_path, learned, name))
            image_id = cursor.lastrowid
        print(f"Image with ID {image_id} successfully inserted.")
        return image_id

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return None

# image 객체 read
def read_image(connection, image_id=None, class_id=None, learned=None):
    try:
        with get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            conditions = []
            values = []

            if image_id is not None:
                conditions.append("image_id = %s")
                values.append(image_id)
            if class_id is not None:
                conditions.append("class_id = %s")
                values.append(class_id)
            if learned is not None:
                conditions.append("learned = %s")
                values.append(learned)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT * FROM image {where_clause};"

            cursor.execute(query, tuple(values))
            rows = cursor.fetchall()

            if not rows:
                print("No image found.")
                return []

            print(f"retrieved {len(rows)} images.")
            return rows
    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return []

# image 객체 update
def update_image(connection, image_id, class_id=None, file_path=None, learned=None, name=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "image", {"id" : image_id}):
                print(f"Image '{image_id}' not found.")
                return False

            updates = []
            values = []

            if class_id is not None:
                updates.append("class_id = %s")
                values.append(class_id)

            if file_path is not None:
                updates.append("file_path = %s")
                values.append(file_path)

            if learned is not None:
                updates.append("learned = %s")
                values.append(learned)

            if name is not None:
                updates.append("name = %s")
                values.append(name)

            if not updates:
                print("No updates.")
                return False

            query = f"UPDATE image SET {', '.join(updates)} WHERE id = %s;"
            values.append(image_id)
            cursor.execute(query, tuple(values))
        print(f"Image with ID {image_id} successfully updated.")
        return True
    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# image 객체 delete
def delete_image(connection, image_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "image", {"id" : image_id}):
                print(f"Image '{image_id}' not found.")
                return False

            cursor.execute("DELETE FROM image WHERE id = %s;", (image_id,))
        print(f"Image with ID {image_id} successfully deleted.")
        return True
    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# image-tag 객체 insert
def insert_image_tag(connection, image_id, tag_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if record_exists(cursor, "image_tag", {"image_id": image_id, "tag_id": tag_id}):
                print(f"Relationship between image ID {image_id} and tag ID {tag_id} already exists.")
                return False

            cursor.execute("""
            INSERT INTO image_tag (image_id, tag_id)
            VALUES (%s, %s);
            """, (image_id, tag_id))
        print(f"Image_Tag with ID {image_id}-{tag_id} successfully inserted.")
        return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# image-tag 객체 read
def read_image_tag(connection, image_id=None, tag_id=None):
    try:
        with get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            conditions = []
            values = []

            if image_id is not None:
                conditions.append("image_id = %s")
                values.append(image_id)
            if tag_id is not None:
                conditions.append("tag_id = %s")
                values.append(tag_id)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT * FROM image_tag WHERE {where_clause};"

            cursor.execute(query, tuple(values))
            rows = cursor.fetchall()

            if not rows:
                print("No image-tag relationships found for the given conditions.")
                return []

            print(f"retrieved {len(rows)} image-tag relationships.")
            return rows

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return []

# image-tag 객체 delete
def delete_image_tag(connection, image_id, tag_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "image_tag", {"image_id": image_id, "tag_id": tag_id}):
                print("No image-tag relationships found for the given conditions.")
                return False

            cursor.execute("""
            DELETE FROM image_tag WHERE image_id = %s AND tag_id = %s;
            """, (image_id, tag_id))

            if cursor.rowcount == 0:
                print(f"Failed to delete relationship for image ID {image_id} and tag ID {tag_id}.")
                return False

        print(f"Deleted relationship between image ID {image_id} and tag ID {tag_id}.")
        return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False


# annotation 객체 insert
def insert_annotation(connection, image_id, x1, y1, x2, y2, label):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "image", {"id" : image_id}):
                print(f"Image with ID {image_id} does not exist.")
                return None

            cursor.execute("""
            INSERT INTO annotation (image_id, x1, y1, x2, y2, label)
            VALUES (%s, %s, %s, %s, %s, %s);
            """, (image_id, x1, y1, x2, y2, label))

            annotation_id = cursor.lastrowid
            print(f"Annotation with ID {annotation_id} successfully inserted.")
            return annotation_id

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# annotation 객체 read
def read_annotation(connection, annotation_id=None, image_id=None):
    try:
        with get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            # 조건별 쿼리 생성
            conditions = []
            values = []

            if annotation_id is not None:
                conditions.append("id = %s")
                values.append(annotation_id)

            if image_id is not None:
                conditions.append("image_id = %s")
                values.append(image_id)

            # WHERE 절 생성
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT * FROM annotation {where_clause};"

            cursor.execute(query, tuple(values))
            rows = cursor.fetchall()

            if not rows:
                print("No annotations found.")
                return []

            print(f"Retrieved {len(rows)} annotations.")
            return rows

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return []

# annotation 객체 update
def update_annotation(connection, x1, y1, x2, y2, annotation_id=None, label=None):
    try:
        with transaction(connection), get_cursor(connection) as cursor:

            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "annotation", {"id" : annotation_id}):
                print(f"Annotation with ID {annotation_id} does not exist.")
                return False

            # 업데이트할 값 준비
            updates = []
            values = []

            if x1 is not None:
                updates.append("x1 = %s")
                values.append(x1)

            if y1 is not None:
                updates.append("y1 = %s")
                values.append(y1)

            if x2 is not None:
                updates.append("x2 = %s")
                values.append(x2)

            if y2 is not None:
                updates.append("y2 = %s")
                values.append(y2)

            if label is not None:
                updates.append("label = %s")
                values.append(label)

            if not updates:
                print("No updates specified.")
                return False

            # 업데이트 쿼리 실행
            query = f"UPDATE annotation SET {', '.join(updates)} WHERE id = %s;"
            values.append(annotation_id)
            cursor.execute(query, tuple(values))

            if cursor.rowcount == 0:
                print(f"No changes made to Annotation with ID {annotation_id}.")
                return False

            print(f"Annotation with ID {annotation_id} updated successfully.")
            return True

    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False

# annotation 객체 delete
def delete_annotation(connection, annotation_id):
    try:
        with transaction(connection), get_cursor(connection) as cursor:
            print(f"Using DB '{database_name}'")

            if not record_exists(cursor, "annotation", {"id" : annotation_id}):
                print(f"Annotation with ID {annotation_id} does not exist.")
                return False

            query = f"DELETE FROM annotation WHERE id = %s;"
            cursor.execute(query, (annotation_id,))
            if cursor.rowcount == 0:
                print(f"No changes made to Annotation with ID {annotation_id}.")
                return False

        print(f"Annotation with ID {annotation_id} deleted successfully.")
        return True
    except mariadb.Error as e:
        print(f"Error occurred: {e}")
        return False
