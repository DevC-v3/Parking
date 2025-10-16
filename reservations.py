from flask import Blueprint, request, jsonify, session
from database import db
from auth import login_required

# Crear Blueprint para reservas
reservations_bp = Blueprint('reservations', __name__)

@reservations_bp.route('/api/reservar', methods=['POST'])
@login_required
def reservar_espacio():
    """API para reservar un espacio de estacionamiento"""
    data = request.get_json()
    space_number = data.get('space_number')
    duration = data.get('duration', 1)
    
    if not space_number:
        return jsonify({'success': False, 'message': 'Número de espacio requerido'}), 400
    
    if db.create_reservation(session['user_id'], space_number, duration):
        return jsonify({
            'success': True, 
            'message': f'Espacio {space_number} reservado por {duration} hora(s)'
        })
    else:
        return jsonify({
            'success': False, 
            'message': f'El espacio {space_number} ya está reservado'
        })

@reservations_bp.route('/api/mis_reservas')
@login_required
def mis_reservas():
    """API para obtener las reservas del usuario actual"""
    reservations = db.get_user_reservations(session['user_id'])
    return jsonify(reservations)

@reservations_bp.route('/api/cancelar_reserva', methods=['POST'])
@login_required
def cancelar_reserva():
    """API para cancelar una reserva"""
    data = request.get_json()
    reservation_id = data.get('reservation_id')
    
    if not reservation_id:
        return jsonify({'success': False, 'message': 'ID de reserva requerido'}), 400
    
    if db.cancel_reservation(reservation_id, session['user_id']):
        return jsonify({'success': True, 'message': 'Reserva cancelada exitosamente'})
    else:
        return jsonify({'success': False, 'message': 'Error al cancelar la reserva'})

@reservations_bp.route('/api/reservas_activas')
def reservas_activas():
    """API pública para obtener espacios reservados (usado por el procesador de video)"""
    reserved_spaces = db.get_active_reservations()
    return jsonify(reserved_spaces)