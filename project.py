import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 800
BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
BUTTON_COLOR = (70, 130, 180)
BUTTON_TEXT_COLOR = (255, 255, 255)
GAP = 20
GRID_SIZE = 5

TILE_COLORS = {
    0: (204, 192, 179), 2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121), 16: (245, 149, 99),
    32: (246, 124, 95), 64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
}
FONT_COLOR = (119, 110, 101)
FONT = pygame.font.Font(None, 48)  # Adjusted font size for better text fit
TIMER_FONT = pygame.font.Font(None, 43)  # Slightly smaller font for buttons

# Game initialization
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Welcome to 2048 AI Game")

class Game2048:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.board = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.add_tile()
        self.add_tile()
        self.game_over = False
        self.paused = False
        self.selected_tile = None

    def add_tile(self):
        empty_tiles = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.board[i][j] == 0]
        if empty_tiles:
            i, j = random.choice(empty_tiles)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def select_tile(self, row, col):
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size and self.board[row][col] != 0:
            self.selected_tile = (row, col)

    def move_selected_tile(self, direction):
        if not self.selected_tile:
            return False
        row, col = self.selected_tile
        if direction == 'UP' and row > 0 and (self.board[row - 1][col] == 0 or self.board[row - 1][col] == self.board[row][col]):
            self.board[row - 1][col] += self.board[row][col]
            self.board[row][col] = 0
            self.selected_tile = (row - 1, col)
            return True
        elif direction == 'DOWN' and row < self.grid_size - 1 and (self.board[row + 1][col] == 0 or self.board[row + 1][col] == self.board[row][col]):
            self.board[row + 1][col] += self.board[row][col]
            self.board[row][col] = 0
            self.selected_tile = (row + 1, col)
            return True
        elif direction == 'LEFT' and col > 0 and (self.board[row][col - 1] == 0 or self.board[row][col - 1] == self.board[row][col]):
            self.board[row][col - 1] += self.board[row][col]
            self.board[row][col] = 0
            self.selected_tile = (row, col - 1)
            return True
        elif direction == 'RIGHT' and col < self.grid_size - 1 and (self.board[row][col + 1] == 0 or self.board[row][col + 1] == self.board[row][col]):
            self.board[row][col + 1] += self.board[row][col]
            self.board[row][col] = 0
            self.selected_tile = (row, col + 1)
            return True
        return False

    def compress_and_merge(self, board):
        moved = False
        new_board = [[0] * self.grid_size for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            position = 0
            for j in range(self.grid_size):
                if board[i][j] != 0:
                    new_board[i][position] = board[i][j]
                    if j != position:
                        moved = True
                    position += 1

        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if new_board[i][j] == new_board[i][j + 1] and new_board[i][j] != 0:
                    new_board[i][j] *= 2
                    new_board[i][j + 1] = 0
                    moved = True

        compressed_board = [[0] * self.grid_size for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            position = 0
            for j in range(self.grid_size):
                if new_board[i][j] != 0:
                    compressed_board[i][position] = new_board[i][j]
                    position += 1

        return compressed_board, moved

    def transpose(self, board):
        return [list(row) for row in zip(*board)]

    def reverse(self, board):
        return [row[::-1] for row in board]

    def check_game_over(self):
        for row in self.board:
            if 0 in row:
                return False
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i < self.grid_size - 1 and self.board[i][j] == self.board[i + 1][j]) or \
                   (j < self.grid_size - 1 and self.board[i][j] == self.board[i][j + 1]):
                    return False
        self.game_over = True
        return True

    def draw(self, screen, time_left):
        screen.fill(BACKGROUND_COLOR)
        tile_size = min(WIDTH, HEIGHT - 200) // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(screen, GRID_COLOR, (j * tile_size + (WIDTH - self.grid_size * tile_size) // 2, i * tile_size + 50, tile_size, tile_size), 1)
                value = self.board[i][j]
                color = TILE_COLORS.get(value, (60, 58, 50))
                pygame.draw.rect(screen, color, (j * tile_size + (WIDTH - self.grid_size * tile_size) // 2, i * tile_size + 50, tile_size, tile_size))
                if value != 0:
                    text = FONT.render(str(value), True, FONT_COLOR)
                    text_rect = text.get_rect(center=(j * tile_size + tile_size // 2 + (WIDTH - self.grid_size * tile_size) // 2, i * tile_size + tile_size // 2 + 50))
                    screen.blit(text, text_rect)
                if self.selected_tile == (i, j):
                    pygame.draw.rect(screen, (255, 0, 0), (j * tile_size + (WIDTH - self.grid_size * tile_size) // 2, i * tile_size + 50, tile_size, tile_size), 3)
        
        timer_text = TIMER_FONT.render(f"Time left: {int(time_left)}s", True, (0, 0, 0))
        timer_rect = timer_text.get_rect(center=(WIDTH // 2, HEIGHT - 120))
        screen.blit(timer_text, timer_rect)

        # Pause and Quit buttons
        button_width = 150
        pause_button = pygame.Rect(WIDTH // 2 - button_width - 10, HEIGHT - 80, button_width, 50)
        quit_button = pygame.Rect(WIDTH // 2 + 10, HEIGHT - 80, button_width, 50)
        self.draw_button(screen, pause_button, "Pause" if not self.paused else "Resume", BUTTON_COLOR, BUTTON_TEXT_COLOR)
        self.draw_button(screen, quit_button, "Quit", BUTTON_COLOR, BUTTON_TEXT_COLOR)

    def draw_button(self, screen, rect, text, color, text_color):
        pygame.draw.rect(screen, color, rect, border_radius=10)
        text_surf = TIMER_FONT.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=10)

    def shuffle_tiles(self):
        non_zero_tiles = [self.board[i][j] for i in range(self.grid_size) for j in range(self.grid_size) if self.board[i][j] != 0]
        random.shuffle(non_zero_tiles)
        empty_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        random.shuffle(empty_positions)
        for (i, j), value in zip(empty_positions, non_zero_tiles):
            self.board[i][j] = value
        for i in range(len(empty_positions) - len(non_zero_tiles)):
            self.board[empty_positions[len(non_zero_tiles) + i][0]][empty_positions[len(non_zero_tiles) + i][1]] = 0

    def draw_shuffling(self, screen):
        screen.fill(BACKGROUND_COLOR)
        shuffling_text = FONT.render("Shuffling...", True, (255, 0, 0))
        shuffling_rect = shuffling_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(shuffling_text, shuffling_rect)
        pygame.display.update()
        self.shuffle_tiles()
        pygame.time.wait(1000)

class AI2048:
    def __init__(self, game):
        self.game = game

    def get_best_move(self):
        best_move = None
        best_score = float('-inf')
        for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            new_board, moved = self.simulate_move(self.game.board, direction)
            if moved:
                score = self.alpha_beta(new_board, 4, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = new_board
        return best_move

    def get_all_possible_moves(self, board):
        moves = []
        for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            new_board, moved = self.simulate_move(board, direction)
            if moved:
                moves.append(new_board)
        return moves

    def simulate_move(self, board, direction):
        new_board = [row[:] for row in board]
        moved = False
        if direction == 'UP':
            new_board = self.game.transpose(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.transpose(new_board)
        elif direction == 'DOWN':
            new_board = self.game.transpose(new_board)
            new_board = self.game.reverse(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.reverse(new_board)
            new_board = self.game.transpose(new_board)
        elif direction == 'LEFT':
            new_board, moved = self.game.compress_and_merge(new_board)
        elif direction == 'RIGHT':
            new_board = self.game.reverse(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.reverse(new_board)
        return new_board, moved

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_terminal_state(board):
            return self.evaluate_board(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_all_possible_moves(board):
                eval = self.alpha_beta(move, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_all_possible_moves(board):
                eval = self.alpha_beta(move, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board):
        empty_tiles = sum(row.count(0) for row in board)
        max_tile = max(max(row) for row in board)
        return empty_tiles - max_tile

    def is_terminal_state(self, board):
        if any(2048 in row for row in board):
            return True
        for i in range(self.game.grid_size):
            for j in range(self.game.grid_size):
                if board[i][j] == 0 or (i < self.game.grid_size - 1 and board[i][j] == board[i + 1][j]) or (j < self.game.grid_size - 1 and board[i][j] == board[i][j + 1]):
                    return False
        return True

class GeneticAlgorithm2048:
    def __init__(self, game):
        self.game = game

    def initialize_population(self, size, board):
        population = []
        non_zero_tiles = [value for row in board for value in row if value != 0]

        for _ in range(size):
            new_board = [[0] * self.game.grid_size for _ in range(self.game.grid_size)]
            random.shuffle(non_zero_tiles)
            index = 0
            for i in range(self.game.grid_size):
                for j in range(self.game.grid_size):
                    if index < len(non_zero_tiles):
                        new_board[i][j] = non_zero_tiles[index]
                        index += 1
            population.append(new_board)
        return population

    def fitness(self, board):
        empty_tiles = sum(row.count(0) for row in board)
        clustering_score = sum(
            abs(board[i][j] - board[i+1][j]) + abs(board[i][j] - board[i][j+1])
            for i in range(len(board)) for j in range(len(board[i]))
            if i + 1 < len(board) and j + 1 < len(board[i])
        )
        return empty_tiles + clustering_score

    def selection(self, population, fitness_scores, num_parents):
        selected = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        return [board for board, score in selected[:num_parents]]

    def crossover(self, parent1, parent2):
        child = []
        for i in range(len(parent1)):
            row = []
            for j in range(len(parent1[i])):
                if random.random() < 0.5:
                    row.append(parent1[i][j])
                else:
                    row.append(parent2[i][j])
            child.append(row)
        return child

    def mutate(self, board, mutation_rate):
        non_zero_tiles = [value for row in board for value in row if value != 0]
        for i in range(len(board)):
            for j in range(len(board[i])):
                if random.random() < mutation_rate and board[i][j] != 0:
                    board[i][j] = random.choice(non_zero_tiles)
        return board

    def genetic_algorithm(self, board, population_size, num_generations, num_parents, mutation_rate):
        population = self.initialize_population(population_size, board)
        for generation in range(num_generations):
            fitness_scores = [self.fitness(board) for board in population]
            parents = self.selection(population, fitness_scores, num_parents)
            new_population = parents[:]
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            population = new_population
        fitness_scores = [self.fitness(board) for board in population]
        best_board = self.selection(population, fitness_scores, 1)[0]
        return best_board

    def get_best_move(self):
        best_board = self.genetic_algorithm(self.game.board, population_size=100, num_generations=50, num_parents=20, mutation_rate=0.1)
        return best_board

class FuzzyAI2048:
    def __init__(self, game):
        self.game = game

    def fuzzy_empty_tiles(self, count):
        if count <= 4:
            return "few"
        elif count <= 8:
            return "moderate"
        else:
            return "many"

    def fuzzy_merge_potential(self, potential):
        if potential <= 15:
            return "low"
        else:
            return "high"

    def fuzzy_player_advantage(self, advantage):
        if advantage <= 50:
            return "low"
        else:
            return "high"

    def fuzzy_move_value(self, empty_tiles, merge_potential, player_advantage):
        if empty_tiles == "few" and merge_potential == "high":
            return "bad"
        if empty_tiles == "many" and merge_potential == "low":
            return "good"
        if player_advantage == "high":
            return "bad"
        return "moderate"

    def get_best_move(self):
        best_move = None
        best_score = float('-inf')
        for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            new_board, moved = self.simulate_move(self.game.board, direction)
            if moved:
                score = self.evaluate_board(new_board)
                if score > best_score:
                    best_score = score
                    best_move = direction
        return best_move

    def simulate_move(self, board, direction):
        new_board = [row[:] for row in board]
        moved = False
        if direction == 'UP':
            new_board = self.game.transpose(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.transpose(new_board)
        elif direction == 'DOWN':
            new_board = self.game.transpose(new_board)
            new_board = self.game.reverse(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.reverse(new_board)
            new_board = self.game.transpose(new_board)
        elif direction == 'LEFT':
            new_board, moved = self.game.compress_and_merge(new_board)
        elif direction == 'RIGHT':
            new_board = self.game.reverse(new_board)
            new_board, moved = self.game.compress_and_merge(new_board)
            new_board = self.game.reverse(new_board)
        return new_board, moved

    def evaluate_board(self, board):
        empty_tiles_count = sum(row.count(0) for row in board)

        merge_potential = 0
        for row in range(self.game.grid_size):
            for col in range(self.game.grid_size):
                if board[row][col] != 0:
                    if row < self.game.grid_size - 1 and board[row][col] == board[row + 1][col]:
                        merge_potential += board[row][col]
                    if col < self.game.grid_size - 1 and board[row][col] == board[row][col + 1]:
                        merge_potential += board[row][col]

        player_advantage = merge_potential * empty_tiles_count

        empty_tiles_fuzzy = self.fuzzy_empty_tiles(empty_tiles_count)
        merge_potential_fuzzy = self.fuzzy_merge_potential(merge_potential)
        player_advantage_fuzzy = self.fuzzy_player_advantage(player_advantage)

        move_value_fuzzy = self.fuzzy_move_value(empty_tiles_fuzzy, merge_potential_fuzzy, player_advantage_fuzzy)

        if move_value_fuzzy == "bad":
            return -1
        elif move_value_fuzzy == "good":
            return 1
        else:
            return 0

def ai_move(game, ai):
    if not game.game_over:
        best_move = ai.get_best_move()
        if best_move:
            game.board = best_move

def ga_move(game, ga):
    if not game.game_over:
        best_board = ga.get_best_move()
        game.board = best_board

def fuzzy_move(game, fuzzy_ai):
    if not game.game_over:
        best_direction = fuzzy_ai.get_best_move()
        if best_direction:
            game.move_selected_tile(best_direction)
            game.add_tile()

# UI functions for mode and difficulty selection
def draw_selection_screen():
    screen.fill(BACKGROUND_COLOR)
    heading_text = FONT.render("Welcome to 2048 AI Game", True, (0, 0, 0))
    heading_rect = heading_text.get_rect(center=(WIDTH // 2, HEIGHT // 12))
    screen.blit(heading_text, heading_rect)

    ai_text = FONT.render("Select Algorithm Mode:", True, (0, 0, 0))
    ai_text_rect = ai_text.get_rect(center=(WIDTH // 2, HEIGHT // 5))
    screen.blit(ai_text, ai_text_rect)

    # AI Mode Buttons (Vertical Alignment)
    alpha_beta_button = pygame.Rect(WIDTH // 4, HEIGHT // 4 + 40, WIDTH // 2, 60)  # Adjusted height for better fit
    fuzzy_button = pygame.Rect(WIDTH // 4, HEIGHT // 4 + 120, WIDTH // 2, 60)
    genetic_button = pygame.Rect(WIDTH // 4, HEIGHT // 4 + 200, WIDTH // 2, 60)

    draw_stylish_button(screen, alpha_beta_button, "Alpha-Beta Pruning", BUTTON_COLOR)
    draw_stylish_button(screen, fuzzy_button, "Fuzzy Logic", BUTTON_COLOR)
    draw_stylish_button(screen, genetic_button, "Genetic Algorithm", BUTTON_COLOR)

    difficulty_text = FONT.render("Select Difficulty Level:", True, (0, 0, 0))
    difficulty_text_rect = difficulty_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
    screen.blit(difficulty_text, difficulty_text_rect)

    # Difficulty Level Buttons (Vertical Alignment)
    easy_button = pygame.Rect(WIDTH // 4, HEIGHT // 2 + 150, WIDTH // 2, 60)
    medium_button = pygame.Rect(WIDTH // 4, HEIGHT // 2 + 230, WIDTH // 2, 60)
    hard_button = pygame.Rect(WIDTH // 4, HEIGHT // 2 + 310, WIDTH // 2, 60)

    draw_stylish_button(screen, easy_button, "Easy: Grid (5x5), Timer (30s)", (0, 135, 0))  # Darker green
    draw_stylish_button(screen, medium_button, "Medium: Grid (4x4), Timer (20s)", (255, 192, 0))  # Darker yellow
    draw_stylish_button(screen, hard_button, "Hard: Grid (3x3), Timer (10s)", (255, 49, 49))  # Darker red

    return alpha_beta_button, fuzzy_button, genetic_button, easy_button, medium_button, hard_button

def draw_stylish_button(screen, rect, text, color):
    pygame.draw.rect(screen, color, rect, border_radius=10)
    text_surf = TIMER_FONT.render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=10)

def draw_countdown(text="Go!"):
    for i in range(3, 0, -1):
        screen.fill(BACKGROUND_COLOR)
        countdown_text = FONT.render(str(i), True, (0, 0, 0))
        countdown_rect = countdown_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(countdown_text, countdown_rect)
        pygame.display.update()
        pygame.time.wait(1000)
    screen.fill(BACKGROUND_COLOR)
    go_text = FONT.render(text, True, (0, 0, 0))
    go_rect = go_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(go_text, go_rect)
    pygame.display.update()
    pygame.time.wait(1000)

# Main game loop with mode and difficulty selection combined
game = None
ai = None
ga = None
fuzzy_ai = None
running = True
clock = pygame.time.Clock()

ai_method = None
difficulty = None
on_selection_screen = True
start_time = 0
time_limit = 30
pause_start_time = 0
paused_time_left = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and on_selection_screen:
            mouse_pos = event.pos
            alpha_beta_button, fuzzy_button, genetic_button, easy_button, medium_button, hard_button = draw_selection_screen()
            if alpha_beta_button.collidepoint(mouse_pos):
                ai_method = 'alpha_beta'
            elif fuzzy_button.collidepoint(mouse_pos):
                ai_method = 'fuzzy'
            elif genetic_button.collidepoint(mouse_pos):
                ai_method = 'genetic'

            if easy_button.collidepoint(mouse_pos):
                difficulty = 'easy'
                GRID_SIZE = 5
                time_limit = 30
            elif medium_button.collidepoint(mouse_pos):
                difficulty = 'medium'
                GRID_SIZE = 4
                time_limit = 20
            elif hard_button.collidepoint(mouse_pos):
                difficulty = 'hard'
                GRID_SIZE = 3
                time_limit = 10

            if ai_method and difficulty:
                game = Game2048(GRID_SIZE)
                if ai_method == 'alpha_beta':
                    ai = AI2048(game)
                elif ai_method == 'genetic':
                    ga = GeneticAlgorithm2048(game)
                elif ai_method == 'fuzzy':
                    fuzzy_ai = FuzzyAI2048(game)
                TILE_SIZE = WIDTH // GRID_SIZE
                on_selection_screen = False
                draw_countdown()
                start_time = time.time()

        if event.type == pygame.KEYDOWN and game and not game.paused:
            if event.key == pygame.K_UP:
                if game.move_selected_tile('UP'):
                    game.add_tile()
            elif event.key == pygame.K_DOWN:
                if game.move_selected_tile('DOWN'):
                    game.add_tile()
            elif event.key == pygame.K_LEFT:
                if game.move_selected_tile('LEFT'):
                    game.add_tile()
            elif event.key == pygame.K_RIGHT:
                if game.move_selected_tile('RIGHT'):
                    game.add_tile()

        if event.type == pygame.MOUSEBUTTONDOWN and game:
            mouse_pos = event.pos
            pause_button = pygame.Rect(WIDTH // 2 - 150 - 10, HEIGHT - 80, 150, 50)
            quit_button = pygame.Rect(WIDTH // 2 + 10, HEIGHT - 80, 150, 50)
            
            if quit_button.collidepoint(mouse_pos):
                running = False
            elif pause_button.collidepoint(mouse_pos):
                if not game.paused:
                    paused_time_left = time_limit - (time.time() - start_time)
                    game.paused = True
                else:
                    start_time = time.time() - (time_limit - paused_time_left)
                    game.paused = False
            else:
                # Calculate the tile size and position correctly
                tile_size = (min(WIDTH, HEIGHT - 200) // GRID_SIZE)
                grid_start_x = (WIDTH - GRID_SIZE * tile_size) // 2
                grid_start_y = 50
        
                if grid_start_x <= mouse_pos[0] <= grid_start_x + GRID_SIZE * tile_size and grid_start_y <= mouse_pos[1] <= grid_start_y + GRID_SIZE * tile_size:
                    col = (mouse_pos[0] - grid_start_x) // tile_size
                    row = (mouse_pos[1] - grid_start_y) // tile_size
                    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                        game.select_tile(row, col)


    if game and not on_selection_screen and not game.paused:
        current_time = time.time()
        time_left = time_limit - (current_time - start_time)
        if time_left <= 0:
            game.paused = True
            game.draw_shuffling(screen)
            draw_countdown("Resume!")
            start_time = time.time()
            game.paused = False

        if game.check_game_over():
            game_over_text = TIMER_FONT.render("Game Over", True, (255, 0, 0))
            game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(game_over_text, game_over_rect)
            pygame.display.update()
            pygame.time.wait(3000)
            running = False

        game.draw(screen, time_left)
        pygame.display.update()
        clock.tick(60)
    elif game and game.paused:
        game.draw(screen, paused_time_left)
        pygame.display.update()
        clock.tick(60)
    else:
        alpha_beta_button, fuzzy_button, genetic_button, easy_button, medium_button, hard_button = draw_selection_screen()
        pygame.display.update()

pygame.quit()