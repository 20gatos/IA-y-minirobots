# =========================================================
# SIMULACIÓN DE INCENDIO FORESTAL CON AUTÓMATAS CELULARES PROBABILÍSTICOS
# =========================================================
"""
Modelo de Autómata Celular Probabilístico para la simulación de propagación
de incendios forestales en un retículo 2D con condición de frontera toroidal
y vecindad de Moore.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ---------------------------------------------------------
# 1. PARÁMETROS GLOBALES DEL MODELO
# ---------------------------------------------------------
N = 100                  # Tamaño del retículo (N x N)
P_CONTAGIO = 0.65        # Probabilidad de contagio a un árbol sano libre
P_IGNICION = 0.00005     # Probabilidad de ignición espontánea (rayo)
DENSIDAD_ARBOLES = 0.80  # Fracción inicial del bosque cubierta por árboles sanos
MAX_PASOS = 300          # Límite máximo de iteraciones temporales

# ---------------------------------------------------------
# 2. DEFINICIÓN DE ESTADOS Y PALETA VISUAL
# ---------------------------------------------------------
VACIO, SANO, FUEGO, CENIZA = 0, 1, 2, 3

# Mapa de colores: Marrón (Vacío), Verde (Sano), Naranja-Rojo (Fuego), Negro (Ceniza)
cmap = ListedColormap(['#8B4513', '#228B22', '#FF4500', '#1a1a1a'])

# Leyenda gráfica para las parcelas
leyenda = [
    Patch(facecolor='#8B4513', label='Vacío (0)'),
    Patch(facecolor='#228B22', label='Sano (1)'),
    Patch(facecolor='#FF4500', label='Fuego (2)'),
    Patch(facecolor='#1a1a1a', label='Ceniza (3)')
]

# ---------------------------------------------------------
# 3. INICIALIZACIÓN DEL BOSQUE
# ---------------------------------------------------------
def inicializar_bosque(N, densidad):
    """
    Genera la matriz inicial del bosque basada en la densidad de vegetación.
    
    Parámetros:
        N (int): Dimensión del retículo (N x N).
        densidad (float): Proporción de celdas iniciales con vegetación sana (0.0 a 1.0).
        
    Retorna:
        np.ndarray: Matriz de tamaño (N, N) con el estado inicial y un foco de fuego central.
    """
    bosque = np.random.choice(
        [VACIO, SANO],
        size=(N, N),
        p=[1 - densidad, densidad]
    )
    # Foco de ignición inicial en el centro del retículo
    bosque[N // 2, N // 2] = FUEGO
    return bosque

# ---------------------------------------------------------
# 4. FUNCIONES DE TRANSICIÓN Y VECINDAD
# ---------------------------------------------------------
def contar_vecinos_fuego(bosque, i, j, N):
    """
    Cuenta el número de celdas vecinas en estado FUEGO en una vecindad de Moore (8 vecinos).
    Aplica condiciones de frontera toroidales (módulos % N).
    """
    vecinos_fuego = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue  # Omitir la celda central
            ni = (i + di) % N   # Conexión toroidal vertical
            nj = (j + dj) % N   # Conexión toroidal horizontal
            if bosque[ni, nj] == FUEGO:
                vecinos_fuego += 1
    return vecinos_fuego

def actualizar(bosque, N, p_contagio, p_ignicion):
    """
    Aplica las reglas de transición probabilísticas de forma síncrona/paralela
    a todas las celdas del autómata celular.
    """
    nuevo_bosque = bosque.copy()

    for i in range(N):
        for j in range(N):
            estado_actual = bosque[i, j]

            if estado_actual == SANO:
                vf = contar_vecinos_fuego(bosque, i, j, N)
                if vf > 0:
                    # Regla de contagio por vecindad
                    if np.random.rand() < p_contagio:
                        nuevo_bosque[i, j] = FUEGO
                else:
                    # Regla de ignición espontánea
                    if np.random.rand() < p_ignicion:
                        nuevo_bosque[i, j] = FUEGO

            elif estado_actual == FUEGO:
                # Regla de consumo: la celda arde y pasa a ceniza en el siguiente paso
                nuevo_bosque[i, j] = CENIZA

            # Los estados VACIO y CENIZA permanecen constantes (absorbentes)

    return nuevo_bosque

# ---------------------------------------------------------
# 5. BUCLE PRINCIPAL DE SIMULACIÓN
# ---------------------------------------------------------
def ejecutar_simulacion():
    bosque = inicializar_bosque(N, DENSIDAD_ARBOLES)
    estado_inicial = bosque.copy()

    # Listas de registro temporal para el análisis de dinámica poblacional
    historial_sanos = []
    historial_fuego = []
    historial_ceniza = []
    historial_vacio = []

    paso = 0
    fuego_activo = True

    while fuego_activo and paso < MAX_PASOS:
        historial_sanos.append(np.sum(bosque == SANO))
        historial_fuego.append(np.sum(bosque == FUEGO))
        historial_ceniza.append(np.sum(bosque == CENIZA))
        historial_vacio.append(np.sum(bosque == VACIO))

        bosque = actualizar(bosque, N, P_CONTAGIO, P_IGNICION)
        paso += 1

        if np.sum(bosque == FUEGO) == 0:
            fuego_activo = False

    estado_final = bosque.copy()

    print("=== RESULTADOS DE LA SIMULACIÓN ===")
    print(f"Pasos ejecutados: {paso}")
    print(f"Árboles sanos iniciales: {np.sum(estado_inicial == SANO)}")
    print(f"Árboles sanos finales:   {np.sum(estado_final == SANO)}")
    print(f"Celdas en ceniza finales: {np.sum(estado_final == CENIZA)}")

    # ---------------------------------------------------------
    # 6. GENERACIÓN DE GRÁFICAS Y VISUALIZACIÓN
    # ---------------------------------------------------------
    # Comparativa Estado Inicial vs Estado Final
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(estado_inicial, cmap=cmap, vmin=0, vmax=3)
    axes[0].set_title("Estado Inicial del Bosque", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(estado_final, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title(f"Estado Final (t = {paso} pasos)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Simulación de Incendio Forestal con AC Probabilístico",
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("/workspace/scratch/comparativa_inicial_final.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Curvas de Propagación Temporal
    tiempo = np.arange(len(historial_sanos))

    plt.figure(figsize=(11, 6))
    plt.plot(tiempo, historial_sanos, label='Sanos', color='#228B22', linewidth=2)
    plt.plot(tiempo, historial_fuego, label='En Fuego', color='#FF4500', linewidth=2)
    plt.plot(tiempo, historial_ceniza, label='Ceniza', color='#1a1a1a', linewidth=2)
    plt.plot(tiempo, historial_vacio, label='Vacío', color='#8B4513', linewidth=2)

    plt.xlabel("Paso de tiempo (t)", fontsize=12)
    plt.ylabel("Número de celdas", fontsize=12)
    plt.title("Evolución Temporal de la Dinámica del Incendio", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/workspace/scratch/curva_propagacion.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Gráficas guardadas con éxito en /workspace/scratch/")

if __name__ == "__main__":
    ejecutar_simulacion()
