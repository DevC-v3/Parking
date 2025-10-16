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
    const parkingLayout = document.getElementById('parkingLayout');
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
}

function actualizarEstadoMapa() {
    fetch('/estado_espacios')
        .then(response => response.json())
        .then(espacios => {
            const totalEspacios = espacios.length;
            let libres = 0;
            
            // Actualizar cada espacio (los IDs en el backend son 0-based)
            espacios.forEach(espacio => {
                const elemento = document.getElementById(`space-${espacio.id + 1}`);
                if (elemento) {
                    if (espacio.ocupado) {
                        elemento.className = 'parking-space-compact ocupado';
                        elemento.querySelector('.space-status-compact').textContent = 'OCUPADO';
                    } else {
                        elemento.className = 'parking-space-compact libre';
                        elemento.querySelector('.space-status-compact').textContent = 'LIBRE';
                        libres++;
                    }
                }
            });
            
            const ocupados = totalEspacios - libres;
            const porcentaje = totalEspacios > 0 ? Math.round((libres / totalEspacios) * 100) : 0;
            
            // Actualizar estadísticas en sidebar
            document.getElementById('totalSide').textContent = totalEspacios;
            document.getElementById('libresSide').textContent = libres;
            document.getElementById('ocupadosSide').textContent = ocupados;
            document.getElementById('porcentajeSide').textContent = porcentaje + '%';
            
            // Actualizar hora
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        })
        .catch(error => console.error('Error:', error));
}

// Inicializar layout y empezar actualizaciones
document.addEventListener('DOMContentLoaded', function() {
    inicializarLayoutMapa();
    setInterval(actualizarEstadoMapa, 1000);
    actualizarEstadoMapa();
});