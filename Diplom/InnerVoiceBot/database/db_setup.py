import psycopg2
import json

try:
    with open('config.json') as f:
        config = json.load(f)

    database_creds_file = config['database_creds_file']

    with open(database_creds_file, 'r') as f:
        database_creds = json.load(f)

except FileNotFoundError:
    print("Ошибка: Файл config.json или database_creds.json не найден.")
    exit(1)
except json.JSONDecodeError:
    print("Ошибка: Не удалось декодировать config.json или database_creds.json. Проверьте синтаксис файла.")
    exit(1)
except KeyError:
    print("Ошибка: В config.json отсутствует ключ 'database_creds_file' или в database_creds.json отсутствуют необходимые ключи.")
    exit(1)


conn = None
cur = None 

try:
    conn = psycopg2.connect(
        host=database_creds['host'],
        user=database_creds['user'],
        password=database_creds['password']
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Проверяем, существует ли база данных
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_creds['database'],))
    exists = cur.fetchone()
    
    # Создаем базу данных, только если она не существует
    if not exists:
        cur.execute(f"CREATE DATABASE {database_creds['database']}")
        print(f"База данных '{database_creds['database']}' создана.")
    else:
        print(f"База данных '{database_creds['database']}' уже существует.")

except psycopg2.Error as e:
    print(f"Ошибка при работе с базой данных: {e}")
    exit(1)
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()

try:
    conn = psycopg2.connect(
        host=database_creds['host'],
        database=database_creds['database'],
        user=database_creds['user'],
        password=database_creds['password']
    )
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS text_messages (
        id SERIAL PRIMARY KEY,
        user_id BIGINT,
        message_id BIGINT,
        text TEXT,
        processed_text TEXT,
        emotion TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS audio_messages (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        user_id BIGINT,
        message_id BIGINT,
        emotion TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS audio_features (
        id SERIAL PRIMARY KEY,
        message_id BIGINT,
        mfccs TEXT, 
        chroma TEXT, 
        mel TEXT, 
        contrast TEXT,
        tonnetz TEXT
    );
    """)

    conn.commit()

except psycopg2.Error as e:
    print(f"Ошибка при работе с базой данных: {e}")
    exit(1)
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()

print("База данных и таблицы успешно созданы (или уже существуют).")