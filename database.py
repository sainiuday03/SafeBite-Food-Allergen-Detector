import hashlib
import hmac
import os
import sqlite3

DB = 'users.db'
ITERATIONS = 200_000


def _hash(password, salt=None):
    """Salted PBKDF2 hash, stored as pbkdf2$salt$hash."""
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2${salt}${digest}"


def _verify(password, stored):
    if stored.startswith('pbkdf2$'):
        salt = stored.split('$')[1]
        return hmac.compare_digest(_hash(password, salt).encode(), stored.encode())
    return hmac.compare_digest(stored.encode(), password.encode())     # old plain-text accounts


def init_db():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            allergies TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def register_user(username, password, allergies_list):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password, allergies) VALUES (?, ?, ?)',
                       (username, _hash(password), ",".join(allergies_list)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute('SELECT password, allergies FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    if not row or not _verify(password, row[0]):
        conn.close()
        return None
    if not row[0].startswith('pbkdf2$'):                              # upgrade an old plain-text password
        cursor.execute('UPDATE users SET password = ? WHERE username = ?', (_hash(password), username))
        conn.commit()
    conn.close()
    return row[1].split(',') if row[1] else []


def update_user_allergies(username, new_allergies_list):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET allergies = ? WHERE username = ?', (",".join(new_allergies_list), username))
    conn.commit()
    conn.close()
    return True
