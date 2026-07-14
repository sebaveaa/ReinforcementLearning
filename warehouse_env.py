import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys

class WarehouseEnv(gym.Env):
    """
    Entorno personalizado de Gymnasium para simular un almacén de 5x5.
    Un robot debe recoger un paquete en una zona de recogida (pickup) 
    y entregarlo en una zona de entrega (drop-off) mientras esquiva 
    un obstáculo móvil que se desplaza de manera estocástica.
    
    Espacio de Acciones (6 discretas):
    - 0: Mover Arriba (Up)
    - 1: Mover Abajo (Down)
    - 2: Mover Izquierda (Left)
    - 3: Mover Derecha (Right)
    - 4: Recoger paquete (Pick up)
    - 5: Entregar paquete (Drop off)
    
    Espacio de Observaciones (MultiDiscrete):
    - [0] Robot X: 0-4
    - [1] Robot Y: 0-4
    - [2] Obstáculo X: 0-4
    - [3] Obstáculo Y: 0-4
    - [4] Recogida (Pickup) X: 0-4
    - [5] Recogida (Pickup) Y: 0-4
    - [6] Entrega (Drop-off) X: 0-4
    - [7] Entrega (Drop-off) Y: 0-4
    - [8] Estado de carga: 0 (sin paquete), 1 (con paquete)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(self, render_mode=None, max_steps=100):
        super().__init__()
        
        self.grid_size = 5
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # 1. Definición del Espacio de Acciones (6 acciones discretas)
        self.action_space = spaces.Discrete(6)
        
        # 2. Definición del Espacio de Observaciones (MultiDiscrete)
        # 5 celdas para cada coordenada (0 a 4) y 2 estados para el paquete (0 o 1)
        self.observation_space = spaces.MultiDiscrete([
            self.grid_size,  # Robot X
            self.grid_size,  # Robot Y
            self.grid_size,  # Obstáculo X
            self.grid_size,  # Obstáculo Y
            self.grid_size,  # Pickup X
            self.grid_size,  # Pickup Y
            self.grid_size,  # Drop-off X
            self.grid_size,  # Drop-off Y
            2                # Estado de carga (0 o 1)
        ])
        
        # Estado interno
        self.robot_pos = np.array([0, 0])
        self.obstacle_pos = np.array([2, 2]) # Posición inicial fija del obstáculo
        self.pickup_pos = np.array([0, 0])
        self.dropoff_pos = np.array([0, 0])
        self.carrying = 0
        self.robot_dir = 0  # Dirección en grados para la boca de Pac-Man (0: Derecha, 90: Arriba, 180: Izquierda, 270: Abajo)
        self._elapsed_steps = 0
        
        # Elementos de renderizado Pygame
        self.window_size = 500  # Ventana de 500x500 píxeles
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        self.pacman_sprite = None
        self.ghost_sprite = None
        self.cherry_sprite = None
        
    def _get_obs(self):
        """Devuelve la observación actual estructurada de acuerdo al espacio."""
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.obstacle_pos[0],
            self.obstacle_pos[1],
            self.pickup_pos[0],
            self.pickup_pos[1],
            self.dropoff_pos[0],
            self.dropoff_pos[1],
            self.carrying
        ], dtype=np.int32)
        
    def _get_info(self):
        """Devuelve información adicional útil para depuración."""
        return {
            "distancia_obstaculo": int(np.linalg.norm(self.robot_pos - self.obstacle_pos, ord=1)),
            "pasos_restantes": self.max_steps - self._elapsed_steps
        }

    def reset(self, seed=None, options=None):
        """Inicializa el entorno a un estado inicial aleatorio estructurado."""
        super().reset(seed=seed)
        
        # Reiniciar contador de pasos
        self._elapsed_steps = 0
        
        # Obstáculo comienza en un punto fijo del almacén (centro: 2, 2)
        self.obstacle_pos = np.array([2, 2], dtype=np.int32)
        
        # Robot inicia típicamente en la esquina (0, 0)
        self.robot_pos = np.array([0, 0], dtype=np.int32)
        
        # Generar zonas de pickup y drop-off de manera aleatoria
        # asegurando que no coincidan entre sí.
        np_random = self.np_random
        
        # Lista de todas las coordenadas posibles del grid
        possible_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        
        # Elegir posiciones distintas para pickup y drop-off
        idx_pickup = np_random.choice(len(possible_positions))
        self.pickup_pos = np.array(possible_positions[idx_pickup], dtype=np.int32)
        
        # Asegurar que drop-off no sea igual a pickup
        remaining_positions = [pos for pos in possible_positions if not np.array_equal(pos, self.pickup_pos)]
        idx_dropoff = np_random.choice(len(remaining_positions))
        self.dropoff_pos = np.array(remaining_positions[idx_dropoff], dtype=np.int32)
        
        # El robot inicia sin paquete y apuntando a la derecha
        self.carrying = 0
        self.robot_dir = 0
        
        # Retornar observación e info
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def step(self, action):
        """Procesa una acción del robot, mueve el obstáculo y calcula el nuevo estado y recompensas."""
        self._elapsed_steps += 1
        
        reward = -1  # Penalización estándar por paso para fomentar eficiencia
        terminated = False
        truncated = False
        info = {}
        
        # Copias de las posiciones antes del movimiento para validaciones
        prev_robot_pos = self.robot_pos.copy()
        
        # 1. Procesar Movimiento / Acción del Robot
        if action == 0:    # Arriba
            self.robot_pos[1] = max(0, self.robot_pos[1] - 1)
            self.robot_dir = 90
        elif action == 1:  # Abajo
            self.robot_pos[1] = min(self.grid_size - 1, self.robot_pos[1] + 1)
            self.robot_dir = 270
        elif action == 2:  # Izquierda
            self.robot_pos[0] = max(0, self.robot_pos[0] - 1)
            self.robot_dir = 180
        elif action == 3:  # Derecha
            self.robot_pos[0] = min(self.grid_size - 1, self.robot_pos[0] + 1)
            self.robot_dir = 0
            
        elif action == 4:  # Recoger Paquete (Pick up)
            # Debe estar sobre la zona de recogida y no estar cargando
            if np.array_equal(self.robot_pos, self.pickup_pos) and self.carrying == 0:
                self.carrying = 1
                reward = 10  # Recompensa positiva por pickup exitoso
            else:
                reward = -10  # Penalización fuerte por intentar recoger inválidamente
                
        elif action == 5:  # Entregar Paquete (Drop off)
            # Debe estar sobre la zona de entrega y estar cargando
            if np.array_equal(self.robot_pos, self.dropoff_pos) and self.carrying == 1:
                self.carrying = 0
                reward = 100  # Gran recompensa por entrega exitosa
                terminated = True  # El episodio termina con éxito
            else:
                reward = -10  # Penalización fuerte por intentar entregar inválidamente

        # Penalización si intentó moverse fuera del tablero (la posición no cambió)
        if action in [0, 1, 2, 3] and np.array_equal(self.robot_pos, prev_robot_pos):
            reward = -2  # Penalización ligera por chocar contra la pared del almacén

        # 2. Movimiento Estocástico del Obstáculo
        # Se mueve de forma aleatoria (20% de probabilidad para cada dirección: Arriba, Abajo, Izquierda, Derecha, Quieto)
        obstacle_action = self.np_random.choice(5)
        if obstacle_action == 0:    # Arriba
            self.obstacle_pos[1] = max(0, self.obstacle_pos[1] - 1)
        elif obstacle_action == 1:  # Abajo
            self.obstacle_pos[1] = min(self.grid_size - 1, self.obstacle_pos[1] + 1)
        elif obstacle_action == 2:  # Izquierda
            self.obstacle_pos[0] = max(0, self.obstacle_pos[0] - 1)
        elif obstacle_action == 3:  # Derecha
            self.obstacle_pos[0] = min(self.grid_size - 1, self.obstacle_pos[0] + 1)
        # Si es 4, se queda en la misma posición (Quieto)

        # 3. Verificación de Colisiones (Robot y Obstáculo)
        if np.array_equal(self.robot_pos, self.obstacle_pos):
            reward = -100  # Penalización masiva por chocar con el obstáculo
            terminated = True  # El episodio finaliza por falla (accidente)
            
        # 4. Verificación de Límite de Pasos (Truncamiento)
        if self._elapsed_steps >= self.max_steps:
            truncated = True

        # Obtener observación e información adicional
        observation = self._get_obs()
        info = self._get_info()
        
        # Renderizado si se activa el modo human
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderiza el estado actual del entorno."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            return self._render_human()

    def _render_ansi(self):
        """Imprime una representación limpia en caracteres ASCII del almacén."""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Colocar elementos fijos en el grid de caracteres
        px, py = self.pickup_pos
        dx, dy = self.dropoff_pos
        ox, oy = self.obstacle_pos
        rx, ry = self.robot_pos
        
        grid[py][px] = "P"
        grid[dy][dx] = "D"
        grid[oy][ox] = "O"
        
        # Colocar robot, si está cargando se representa diferente
        robot_char = "R*" if self.carrying == 1 else "R"
        grid[ry][rx] = robot_char
        
        # Construir la cadena a imprimir
        out = "\n" + "-" * 25 + "\n"
        for row in grid:
            out += " | ".join(f"{cell:2s}" for cell in row) + "\n"
        out += "-" * 25 + "\n"
        print(out)
        return out

    def _render_human(self):
        """Dibuja una interfaz gráfica premium usando Pygame."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            # Nombre de la ventana
            pygame.display.set_caption("Warehouse Automated Grid 5x5 - AI Lab")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self._init_sprites()
            
        # Procesar cola de eventos de Pygame para mantener la ventana interactiva y responsiva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Paleta de Colores Moderna (Estética Premium Dark Mode)
        color_bg = (30, 30, 36)          # Gris carbón oscuro
        color_grid = (45, 45, 56)        # Gris intermedio para la cuadrícula
        color_pickup = (0, 180, 216)     # Azul brillante (cargando/recogida)
        color_dropoff = (46, 196, 182)   # Verde azulado esmeralda (entrega)
        color_obstacle = (230, 57, 70)   # Naranja rojizo (peligro/obstáculo)
        color_robot = (255, 183, 3)      # Amarillo dorado (robot)
        color_package = (160, 108, 213)  # Púrpura orquídea (paquete)
        
        # Limpiar pantalla con color de fondo
        self.window.fill(color_bg)
        
        # 1. Dibujar zonas de Pickup y Dropoff primero (bajo los agentes)
        # Pickup Zone
        pickup_rect = pygame.Rect(
            self.pickup_pos[0] * self.cell_size,
            self.pickup_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.window, color_pickup, pickup_rect)
        # Borde sutil interno
        pygame.draw.rect(self.window, (255, 255, 255), pickup_rect, 2)
        
        # Dropoff Zone
        dropoff_rect = pygame.Rect(
            self.dropoff_pos[0] * self.cell_size,
            self.dropoff_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.window, color_dropoff, dropoff_rect)
        pygame.draw.rect(self.window, (255, 255, 255), dropoff_rect, 2)

        # 2. Dibujar líneas de cuadrícula para una estética limpia
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.window,
                color_grid,
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size),
                2
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                self.window,
                color_grid,
                (0, y * self.cell_size),
                (self.window_size, y * self.cell_size),
                2
            )

        # 3. Colocar texto descriptivo en las zonas (P para pickup, D para dropoff)
        try:
            font = pygame.font.SysFont("Outfit", 28, bold=True)
        except:
            font = pygame.font.Font(None, 36)
            
        txt_p = font.render("PICKUP", True, (255, 255, 255))
        txt_d = font.render("DROP-OFF", True, (255, 255, 255))
        
        # Centrar texto en celdas
        self.window.blit(txt_p, (pickup_rect.centerx - txt_p.get_width() // 2, pickup_rect.centery - txt_p.get_height() // 2))
        self.window.blit(txt_d, (dropoff_rect.centerx - txt_d.get_width() // 2, dropoff_rect.centery - txt_d.get_height() // 2))

        # 4. Dibujar el Obstáculo (Fantasma pixel-art)
        obs_center_x = self.obstacle_pos[0] * self.cell_size + self.cell_size // 2
        obs_center_y = self.obstacle_pos[1] * self.cell_size + self.cell_size // 2
        ghost_rect = self.ghost_sprite.get_rect(center=(obs_center_x, obs_center_y))
        self.window.blit(self.ghost_sprite, ghost_rect)

        # 5. Dibujar al Robot (Pac-Man pixel-art rotado en la dirección de su movimiento)
        robot_center_x = self.robot_pos[0] * self.cell_size + self.cell_size // 2
        robot_center_y = self.robot_pos[1] * self.cell_size + self.cell_size // 2
        
        rotated_pacman = pygame.transform.rotate(self.pacman_sprite, self.robot_dir)
        pacman_rect = rotated_pacman.get_rect(center=(robot_center_x, robot_center_y))
        self.window.blit(rotated_pacman, pacman_rect)
        
        # Si lleva el paquete, dibujamos la cereza encima
        if self.carrying == 1:
            cherry_rect = self.cherry_sprite.get_rect(center=(robot_center_x + int(self.cell_size * 0.2), robot_center_y - int(self.cell_size * 0.2)))
            self.window.blit(self.cherry_sprite, cherry_rect)

        # Actualizar display
        pygame.event.pump()
        pygame.display.flip()
        
        # Limitar framerate según la metadata del entorno
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Cierra los recursos gráficos abiertos de Pygame."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _init_sprites(self):
        """Genera superficies en memoria con pixel art retro para Pac-Man, el Fantasma y las Cerezas."""
        # Paleta de colores para los sprites
        # Usamos (0, 0, 0) como color transparente y llamamos a set_colorkey((0,0,0))
        palette = {
            ".": (0, 0, 0),        # Transparente
            "Y": (255, 238, 0),    # Amarillo Pac-Man
            "K": (0, 0, 0),        # Negro Ojos
            "P": (255, 150, 180),  # Rosa tierno para el rubor (blush)
            "R": (255, 0, 0),      # Rojo Fantasma (Blinky)
            "W": (255, 255, 255),  # Blanco
            "B": (33, 150, 243),   # Azul
            "G": (76, 175, 80),    # Verde Cereza
            "C": (217, 4, 41),     # Rojo Cereza
        }
        
        # Patrón de Pac-Man (16x16) mirando a la derecha
        # Tiene una forma más redonda, un ojito tierno de 2x2 y rubor rosa en la mejilla
        pacman_pattern = [
            ".....YYYYYY.....",
            "...YYYYYYYYYY...",
            "..YYYYYYYYYYYY..",
            ".YYYYYKKYYYYYYY.",
            ".YYYYYKKPYYYYYY.",
            "YYYYYYYPPYYY....",
            "YYYYYYYYYY......",
            "YYYYYYYY........",
            "YYYYYYYYYY......",
            "YYYYYYYYYYYY....",
            ".YYYYYYYYYYYYY..",
            ".YYYYYYYYYYYYY..",
            "..YYYYYYYYYYYY..",
            "...YYYYYYYYYY...",
            ".....YYYYYY.....",
            "................"
        ]
        
        # Patrón de Fantasma (16x16)
        # Ojos redondos grandes y separados para un look mucho más tierno
        ghost_pattern = [
            ".....RRRRRR.....",
            "...RRRRRRRRRR...",
            "..RRRRRRRRRRRR..",
            ".RRRRRRRRRRRRRR.",
            ".RRWWWRRRRWWWRR.",
            "RWWWWWRRWWWWWRR",
            "RWWBBWRRWWBBWRR",
            "RWWBBWRRWWBBWRR",
            ".RRWWWRRRRWWWRR.",
            ".RRRRRRRRRRRRRR.",
            "RRRRRRRRRRRRRRRR",
            "RRRRRRRRRRRRRRRR",
            "RRRRRRRRRRRRRRRR",
            "RR.RRR..RRR.RR.R",
            "R...R....R...R..",
            "................"
        ]
        
        # Patrón de Cereza (16x16)
        cherry_pattern = [
            "........GG......",
            ".......G........",
            "......G.........",
            "...GGGGGGGG.....",
            "..G...G....G....",
            ".G....G.....G...",
            ".G.....G....G...",
            "....CC....CC....",
            "...CCCC..CCCC...",
            "..CCWCCCCCWCCC..",
            "..CCWCCCCCWCCC..",
            "..CCCCCCCCCCCC..",
            "...CCCC..CCCC...",
            "....CC....CC....",
            "................",
            "................"
        ]
        
        # Función interna para convertir patrón de caracteres en pygame.Surface
        def make_surface(pattern):
            surf = pygame.Surface((16, 16))
            surf.fill((0, 0, 0)) # Fondo negro para usar de colorkey
            surf.set_colorkey((0, 0, 0))
            for y, row in enumerate(pattern):
                for x, char in enumerate(row):
                    color = palette.get(char, (0, 0, 0))
                    if char != ".":
                        surf.set_at((x, y), color)
            return surf

        # Crear y escalar los sprites
        sprite_size = int(self.cell_size * 0.75)
        raw_pacman = make_surface(pacman_pattern)
        raw_ghost = make_surface(ghost_pattern)
        raw_cherry = make_surface(cherry_pattern)
        
        self.pacman_sprite = pygame.transform.scale(raw_pacman, (sprite_size, sprite_size))
        self.ghost_sprite = pygame.transform.scale(raw_ghost, (sprite_size, sprite_size))
        self.cherry_sprite = pygame.transform.scale(raw_cherry, (int(sprite_size * 0.8), int(sprite_size * 0.8)))
