import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash

# Obtener el directorio base de la aplicación
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración de SQLAlchemy
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con reservas
    reservations = relationship("Reservation", back_populates="user")

class Reservation(Base):
    __tablename__ = 'reservations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    space_number = Column(Integer, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con usuario
    user = relationship("User", back_populates="reservations")

class Database:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = f"sqlite:///{os.path.join(BASE_DIR, 'parking.db')}"
        
        self.engine = create_engine(db_path, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Crear tablas de usuarios y reservas si no existen"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_db(self) -> Session:
        """Obtener sesión de base de datos"""
        db = self.SessionLocal()
        try:
            return db
        finally:
            db.close()
    
    # ==================== USUARIOS ====================
    
    def create_user(self, username, password):
        """Crear nuevo usuario en la base de datos"""
        db = self.get_db()
        try:
            hashed_pw = generate_password_hash(password)
            user = User(username=username, password=hashed_pw)
            db.add(user)
            db.commit()
            db.refresh(user)
            return True
        except IntegrityError:
            db.rollback()
            return False
        finally:
            db.close()
    
    def authenticate_user(self, username, password):
        """Verificar credenciales de usuario"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and check_password_hash(user.password, password):
                return user.id  # user_id
            return None
        finally:
            db.close()
    
    def get_user_by_id(self, user_id):
        """Obtener información de usuario por ID"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return {'id': user.id, 'username': user.username} if user else None
        finally:
            db.close()
    
    # ==================== RESERVAS ====================
    
    def create_reservation(self, user_id, space_number, duration_hours=1):
        """Crear nueva reserva de espacio"""
        db = self.get_db()
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            
            # Verificar si el espacio ya está reservado
            existing_reservation = db.query(Reservation).filter(
                Reservation.space_number == space_number,
                Reservation.status == 'active',
                Reservation.end_time > start_time
            ).first()
            
            if existing_reservation:
                return False  # Espacio ya reservado
            
            # Crear la reserva
            reservation = Reservation(
                user_id=user_id,
                space_number=space_number,
                start_time=start_time,
                end_time=end_time
            )
            db.add(reservation)
            db.commit()
            return True
        finally:
            db.close()
    
    def get_active_reservations(self):
        """Obtener lista de espacios actualmente reservados"""
        db = self.get_db()
        try:
            reservations = db.query(Reservation.space_number).filter(
                Reservation.status == 'active',
                Reservation.end_time > datetime.now()
            ).all()
            return [row[0] for row in reservations]
        finally:
            db.close()
    
    def get_user_reservations(self, user_id):
        """Obtener todas las reservas de un usuario"""
        db = self.get_db()
        try:
            reservations = db.query(Reservation).filter(
                Reservation.user_id == user_id
            ).order_by(Reservation.created_at.desc()).all()
            
            result = []
            for res in reservations:
                result.append({
                    'id': res.id,
                    'space_number': res.space_number,
                    'start_time': res.start_time,
                    'end_time': res.end_time,
                    'status': res.status
                })
            return result
        finally:
            db.close()
    
    def cancel_reservation(self, reservation_id, user_id):
        """Cancelar una reserva (solo si pertenece al usuario)"""
        db = self.get_db()
        try:
            reservation = db.query(Reservation).filter(
                Reservation.id == reservation_id,
                Reservation.user_id == user_id
            ).first()
            
            if reservation:
                reservation.status = 'cancelled'
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    def cleanup_expired_reservations(self):
        """Limpiar reservas expiradas (puede ejecutarse periódicamente)"""
        db = self.get_db()
        try:
            result = db.query(Reservation).filter(
                Reservation.status == 'active',
                Reservation.end_time < datetime.now()
            ).update({'status': 'expired'})
            
            db.commit()
            return result
        finally:
            db.close()
#admin
    def get_all_reservations(self):
        """Obtener todas las reservas (solo para admin)"""
        db = self.get_db()
        try:
            reservations = db.query(Reservation).order_by(Reservation.created_at.desc()).all()
            result = []
            for res in reservations:
                user = db.query(User).filter(User.id == res.user_id).first()
                result.append({
                    'id': res.id,
                    'user_id': res.user_id,
                    'username': user.username if user else 'N/A',
                    'space_number': res.space_number,
                    'start_time': res.start_time.isoformat() if res.start_time else None,
                    'end_time': res.end_time.isoformat() if res.end_time else None,
                    'status': res.status,
                    'created_at': res.created_at.isoformat() if res.created_at else None
                })
            return result
        finally:
            db.close()
    
    def get_all_users(self):
        """Obtener todos los usuarios (solo para admin)"""
        db = self.get_db()
        try:
            users = db.query(User).order_by(User.created_at.desc()).all()
            result = []
            for user in users:
                result.append({
                    'id': user.id,
                    'username': user.username,
                    'admin': user.admin,
                    'created_at': user.created_at.isoformat() if user.created_at else None
                })
            return result
        finally:
            db.close()
    
    def get_user_admin_status(self, user_id):
        """Obtener el estado de admin de un usuario"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user.admin if user else False
        finally:
            db.close()

# Instancia global de la base de datos
db = Database()