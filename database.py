import sqlite3
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

class Database:
    def __init__(self, db_path='parking.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        """Crear tablas de usuarios y reservas si no existen"""
        cursor = self.conn.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de reservas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                space_number INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.conn.commit()
    
    # ==================== USUARIOS ====================
    
    def create_user(self, username, password):
        """Crear nuevo usuario en la base de datos"""
        try:
            cursor = self.conn.cursor()
            hashed_pw = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (username, password) VALUES (?, ?)', 
                (username, hashed_pw)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username, password):
        """Verificar credenciales de usuario"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, password FROM users WHERE username = ?', 
            (username,)
        )
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            return user[0]  # user_id
        return None
    
    def get_user_by_id(self, user_id):
        """Obtener información de usuario por ID"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, username FROM users WHERE id = ?', 
            (user_id,)
        )
        user = cursor.fetchone()
        return {'id': user[0], 'username': user[1]} if user else None
    
    # ==================== RESERVAS ====================
    
    def create_reservation(self, user_id, space_number, duration_hours=1):
        """Crear nueva reserva de espacio"""
        cursor = self.conn.cursor()
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Verificar si el espacio ya está reservado
        cursor.execute('''
            SELECT id FROM reservations 
            WHERE space_number = ? AND status = 'active' AND end_time > ?
        ''', (space_number, start_time))
        
        if cursor.fetchone():
            return False  # Espacio ya reservado
        
        # Crear la reserva
        cursor.execute('''
            INSERT INTO reservations (user_id, space_number, start_time, end_time)
            VALUES (?, ?, ?, ?)
        ''', (user_id, space_number, start_time, end_time))
        
        self.conn.commit()
        return True
    
    def get_active_reservations(self):
        """Obtener lista de espacios actualmente reservados"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT space_number FROM reservations 
            WHERE status = 'active' AND end_time > ?
        ''', (datetime.now(),))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_user_reservations(self, user_id):
        """Obtener todas las reservas de un usuario"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, space_number, start_time, end_time, status 
            FROM reservations 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        reservations = []
        for res in cursor.fetchall():
            reservations.append({
                'id': res[0],
                'space_number': res[1],
                'start_time': res[2],
                'end_time': res[3],
                'status': res[4]
            })
        
        return reservations
    
    def cancel_reservation(self, reservation_id, user_id):
        """Cancelar una reserva (solo si pertenece al usuario)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE reservations SET status = 'cancelled' 
            WHERE id = ? AND user_id = ?
        ''', (reservation_id, user_id))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def cleanup_expired_reservations(self):
        """Limpiar reservas expiradas (puede ejecutarse periódicamente)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE reservations SET status = 'expired' 
            WHERE status = 'active' AND end_time < ?
        ''', (datetime.now(),))
        
        self.conn.commit()
        return cursor.rowcount

# Instancia global de la base de datos
db = Database()