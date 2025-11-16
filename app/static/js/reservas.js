// Configuración del layout
const layoutConfig = [
    { 
        tipo: 'espacios', 
        numeros: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
    },
    { 
        tipo: 'espacios', 
        numeros: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 
    },
    { 
        tipo: 'pista', 
        texto: 'PISTA' 
    },
    { 
        tipo: 'espacios', 
        numeros: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] 
    },
    { 
        tipo: 'espacios', 
        numeros: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46] 
    },
    { 
        tipo: 'pista', 
        texto: 'PISTA' 
    },
    { 
        tipo: 'espacios', 
        numeros: [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58] 
    },
    { 
        tipo: 'espacios', 
        numeros: [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69] 
    }
];

let selectedSpace = null;
let espacioStates = {};

function inicializarLayoutReservas() {
    const parkingLayout = document.getElementById('parkingLayoutReservas');
    parkingLayout.innerHTML = '';
    
    layoutConfig.forEach((columna, index) => {
        const columnaElement = document.createElement('div');
        columnaElement.className = 'columna-reservas';
        columnaElement.id = `columna-reservas-${index + 1}`;
        
        if (columna.tipo === 'pista') {
            const pista = document.createElement('div');
            pista.className = 'parking-space-reservas pista-reservas';
            pista.innerHTML = `<div class="pista-text-reservas">${columna.texto}</div>`;
            columnaElement.appendChild(pista);
        } else {
            columna.numeros.forEach(numeroEspacio => {
                const espacio = document.createElement('div');
                espacio.className = 'parking-space-reservas';
                espacio.id = `space-reserva-${numeroEspacio}`;
                espacio.dataset.spaceNumber = numeroEspacio;
                espacio.innerHTML = `
                    <div class="space-number-reservas">${numeroEspacio}</div>
                    <div class="space-status-reservas">---</div>
                `;
                columnaElement.appendChild(espacio);
            });
        }
        
        parkingLayout.appendChild(columnaElement);
    });
}

function setupEventHandlers() {
    // Click en espacios
    document.addEventListener('click', function(e) {
        const spaceElement = e.target.closest('.parking-space-reservas');
        if (spaceElement && !spaceElement.classList.contains('pista-reservas')) {
            const spaceNumber = parseInt(spaceElement.dataset.spaceNumber);
            selectSpace(spaceNumber, spaceElement);
        }
    });

    // Cambio en duración
    document.querySelectorAll('input[name="duration"]').forEach(radio => {
        radio.addEventListener('change', updateReservationSummary);
    });

    // Formulario de reserva
    document.getElementById('reservationForm').addEventListener('submit', function(e) {
        e.preventDefault();
        makeReservation();
    });

    // Actualizar hora actual
    setInterval(updateCurrentTime, 1000);
    updateCurrentTime();
}

function selectSpace(spaceNumber, spaceElement) {
    const estado = espacioStates[spaceNumber];
    
    if (!estado || estado.reservado || estado.ocupado) {
        showAlert('Este espacio no está disponible para reservar', 'warning');
        return;
    }

    // Remover selección anterior
    document.querySelectorAll('.parking-space-reservas.selected').forEach(el => {
        el.classList.remove('selected');
    });

    // Seleccionar nuevo espacio
    spaceElement.classList.add('selected');
    selectedSpace = spaceNumber;

    // Mostrar información del espacio seleccionado
    document.getElementById('noSpaceSelected').classList.add('d-none');
    document.getElementById('spaceSelected').classList.remove('d-none');
    
    document.getElementById('selectedSpaceNumber').textContent = spaceNumber;
    document.getElementById('previewSpaceNumber').textContent = spaceNumber;
    document.getElementById('summarySpace').textContent = spaceNumber;

    // Habilitar botón de reserva
    document.getElementById('btnReservar').disabled = false;

    // Actualizar resumen
    updateReservationSummary();
}

function updateReservationSummary() {
    if (!selectedSpace) return;

    const duration = document.querySelector('input[name="duration"]:checked').value;
    const endTime = new Date();
    endTime.setHours(endTime.getHours() + parseInt(duration));

    document.getElementById('summaryDuration').textContent = duration + ' hora(s)';
    document.getElementById('summaryEndTime').textContent = endTime.toLocaleTimeString();
}

function updateCurrentTime() {
    document.getElementById('currentTime').textContent = new Date().toLocaleTimeString();
}

function makeReservation() {
    if (!selectedSpace) return;

    const duration = document.querySelector('input[name="duration"]:checked').value;
    
    fetch('/api/reservar', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            space_number: selectedSpace,
            duration: parseInt(duration)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showConfirmationModal(selectedSpace, duration);
            loadUserReservations();
            actualizarEstadoReservas();
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error al realizar la reserva', 'error');
    });
}

function showConfirmationModal(spaceNumber, duration) {
    const endTime = new Date();
    endTime.setHours(endTime.getHours() + parseInt(duration));
    
    document.getElementById('confirmedSpace').textContent = spaceNumber;
    document.getElementById('confirmedDuration').textContent = duration + ' hora(s)';
    document.getElementById('confirmedEndTime').textContent = endTime.toLocaleTimeString();
    
    new bootstrap.Modal(document.getElementById('confirmationModal')).show();
}

function actualizarEstadoReservas() {
    fetch('/estado_espacios')
        .then(response => response.json())
        .then(espacios => {
            // Guardar estados para verificación
            espacios.forEach(espacio => {
                espacioStates[espacio.id + 1] = {
                    ocupado: espacio.ocupado,
                    reservado: espacio.reservado
                };
            });

            // Actualizar visualización
            espacios.forEach(espacio => {
                const elemento = document.getElementById(`space-reserva-${espacio.id + 1}`);
                if (elemento) {
                    if (espacio.reservado) {
                        elemento.className = 'parking-space-reservas reservado';
                        elemento.querySelector('.space-status-reservas').textContent = 'RESERVADO';
                    } else if (espacio.ocupado) {
                        elemento.className = 'parking-space-reservas ocupado';
                        elemento.querySelector('.space-status-reservas').textContent = 'OCUPADO';
                    } else {
                        elemento.className = 'parking-space-reservas libre';
                        elemento.querySelector('.space-status-reservas').textContent = 'DISPONIBLE';
                    }
                }
            });
        })
        .catch(error => console.error('Error:', error));
}

function loadUserReservations() {
    fetch('/api/mis_reservas')
        .then(response => response.json())
        .then(reservations => {
            const container = document.getElementById('reservationsList');
            
            if (reservations.length === 0) {
                container.innerHTML = '<p class="text-center text-muted">No tienes reservas activas</p>';
                return;
            }

            let html = '';
            reservations.forEach(res => {
                const startTime = new Date(res.start_time).toLocaleString();
                const endTime = new Date(res.end_time).toLocaleString();
                const isActive = res.status === 'active' && new Date(res.end_time) > new Date();
                
                if (isActive) {
                    html += `
                        <div class="reservation-item-reserva active mb-2">
                            <div class="reservation-space-reserva">Espacio ${res.space_number}</div>
                            <div class="reservation-time-reserva">Hasta: ${endTime}</div>
                        </div>
                    `;
                }
            });

            container.innerHTML = html || '<p class="text-center text-muted">No tienes reservas activas</p>';
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('reservationsList').innerHTML = 
                '<p class="text-center text-danger">Error al cargar reservas</p>';
        });
}

function showAlert(message, type) {
    const alertClass = type === 'error' ? 'alert-danger' : 'alert-warning';
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    alert.style.top = '20px';
    alert.style.right = '20px';
    alert.style.zIndex = '9999';
    alert.style.minWidth = '300px';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alert);
    
    setTimeout(() => {
        if (alert.parentElement) {
            alert.remove();
        }
    }, 5000);
}

// Inicializar
document.addEventListener('DOMContentLoaded', function() {
    inicializarLayoutReservas();
    setupEventHandlers();
    setInterval(actualizarEstadoReservas, 2000);
    setInterval(loadUserReservations, 5000);
    actualizarEstadoReservas();
    loadUserReservations();
});