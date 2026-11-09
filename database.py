import hashlib
import hmac
import os
import secrets
import sqlite3
import time

DB = 'users.db'
ITERATIONS = 200_000
SESSION_DAYS = 7


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
    cursor.execute('CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, username TEXT NOT NULL, expires REAL NOT NULL)')
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



# --- login sessions: a random token kept in the page URL lets the user stay logged in after a reload ---
def create_session(username):
    token = secrets.token_urlsafe(32)
    conn = sqlite3.connect(DB)
    conn.execute('DELETE FROM sessions WHERE expires < ?', (time.time(),))
    conn.execute('INSERT INTO sessions (token, username, expires) VALUES (?, ?, ?)',
                 (token, username, time.time() + SESSION_DAYS * 86400))
    conn.commit()
    conn.close()
    return token


def get_session_user(token):
    """Returns (username, allergies) for a valid, unexpired token, else None."""
    if not token:
        return None
    conn = sqlite3.connect(DB)
    row = conn.execute('SELECT u.username, u.allergies FROM sessions s JOIN users u ON u.username = s.username '
                       'WHERE s.token = ? AND s.expires > ?', (token, time.time())).fetchone()
    conn.close()
    return (row[0], row[1].split(',') if row[1] else []) if row else None


def delete_session(token):
    conn = sqlite3.connect(DB)
    conn.execute('DELETE FROM sessions WHERE token = ?', (token,))
    conn.commit()
    conn.close()
