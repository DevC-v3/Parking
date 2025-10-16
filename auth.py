from flask import Blueprint, request, session, redirect, url_for, flash, render_template
from database import db

# Crear Blueprint para autenticación
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Manejar inicio de sesión de usuarios"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_id = db.authenticate_user(username, password)
        if user_id:
            session['user_id'] = user_id
            session['username'] = username
            flash('¡Inicio de sesión exitoso!', 'success')
            return redirect(url_for('mapa'))
        else:
            flash('Usuario o contraseña incorrectos', 'error')
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Manejar registro de nuevos usuarios"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if db.create_user(username, password):
            flash('¡Usuario registrado exitosamente! Ahora puedes iniciar sesión.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('El nombre de usuario ya existe', 'error')
    
    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    """Cerrar sesión del usuario"""
    session.clear()
    flash('¡Sesión cerrada exitosamente!', 'success')
    return redirect(url_for('index'))

def login_required(f):
    """Decorador para proteger rutas que requieren autenticación"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Por favor inicia sesión para acceder a esta página', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    
    return decorated_function