// Configuración del layout - 8 COLUMNAS (6 espacios + 2 pistas)
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

function inicializarLayoutMapa() {
    console.log("Inicializando mapa del estacionamiento...");
    const parkingLayout = document.getElementById('parkingLayout');
    
    if (!parkingLayout) {
        console.error("No se encontró el elemento parkingLayout");
        return;
    }
    
    parkingLayout.innerHTML = ''; // Limpiar
    
    layoutConfig.forEach((columna, index) => {
        const columnaElement = document.createElement('div');
        columnaElement.className = 'columna-compact';
        columnaElement.id = `columna-${index + 1}`;
        
        if (columna.tipo === 'pista') {
            const pista = document.createElement('div');
            pista.className = 'parking-space-compact pista-compact';
            pista.innerHTML = `<div class="pista-text-compact">${columna.texto}</div>`;
            columnaElement.appendChild(pista);
        } else {
            // Crear espacios para esta columna
            columna.numeros.forEach(numeroEspacio => {
                const espacio = document.createElement('div');
                espacio.className = 'parking-space-compact';
                espacio.id = `space-${numeroEspacio}`;
                espacio.innerHTML = `
                    <div class="space-number-compact">${numeroEspacio}</div>
                    <div class="space-status-compact">---</div>
                `;
                columnaElement.appendChild(espacio);
            });
        }
        
        parkingLayout.appendChild(columnaElement);
    });
    
    console.log("Mapa inicializado correctamente");
}

function actualizarEstadoMapa() {
    fetch('/estado_espacios')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            return response.json();
        })
        .then(espacios => {
            const totalEspacios = espacios.length;
            let libres = 0;
            let ocupados = 0;
            let reservados = 0;
            
            espacios.forEach(espacio => {
                const elemento = document.getElementById(`space-${espacio.id + 1}`);
                if (elemento) {
                    if (espacio.reservado) {
                        elemento.className = 'parking-space-compact reservado';
                        elemento.querySelector('.space-status-compact').textContent = 'RESERVADO';
                        reservados++;
                    } else if (espacio.ocupado) {
                        elemento.className = 'parking-space-compact ocupado';
                        elemento.querySelector('.space-status-compact').textContent = 'OCUPADO';
                        ocupados++;
                    } else {
                        elemento.className = 'parking-space-compact libre';
                        elemento.querySelector('.space-status-compact').textContent = 'LIBRE';
                        libres++;
                    }
                }
            });
            
            const porcentaje = totalEspacios > 0 ? Math.round((libres / totalEspacios) * 100) : 0;
            
            // Actualizar estadísticas
            document.getElementById('totalSide').textContent = totalEspacios;
            document.getElementById('libresSide').textContent = libres;
            document.getElementById('ocupadosSide').textContent = ocupados;
            document.getElementById('porcentajeSide').textContent = porcentaje + '%';
            
            // Actualizar hora
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => {
            console.error('Error al actualizar estado:', error);
            // Mostrar datos de prueba si hay error
            document.getElementById('totalSide').textContent = '69';
            document.getElementById('libresSide').textContent = '?';
            document.getElementById('ocupadosSide').textContent = '?';
            document.getElementById('porcentajeSide').textContent = '?%';
        });
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM cargado, inicializando mapa...");
    inicializarLayoutMapa();
    
    // Actualizar cada 2 segundos
    setInterval(actualizarEstadoMapa, 2000);
    
    // Primera actualización inmediata
    setTimeout(actualizarEstadoMapa, 1000);
});