import pygame
import os
import random
import math
import sys
import neat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

pygame.init()

SCREEN_HEIGHT = 760
SCREEN_WIDTH = 1280
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Images", "DinoRun1.png")),
           pygame.image.load(os.path.join("Images", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join("Images", "DinoJump.png"))

DUCKING = [pygame.image.load(os.path.join("Images", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Images", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Images", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Images", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Images", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Images", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Images", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Images", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Images", "Bird1.png")),
        pygame.image.load(os.path.join("Images", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Images", "Cloud.png"))

BG = pygame.image.load(os.path.join("Images", "Track.png"))

FONT = pygame.font.Font('freesansbold.ttf', 20)

generation_stats = {
    'max_fitness': [],
    'avg_fitness': [],
    'max_score': [],
    'avg_score': [],
    'species_count': [],
    'generation': []
}

current_gen_scores = []
current_gen_fitnesses = []


# Defining Playable Character
class Dinosaur:
    BASE_X_POS = 80
    X_SPACING = 5 
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self, dino_index=0, img=RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.dino_duck = False
        self.jump_vel = self.JUMP_VEL
        self.dino_index = dino_index

        self.X_POS = self.BASE_X_POS + (self.dino_index * self.X_SPACING)

        self.rect = pygame.Rect(self.X_POS, self.Y_POS,
                                img.get_width(), img.get_height())
        self.color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        self.step_index = 0
        self.duck_img = DUCKING

    # Updates the dinosaur's state and animation based on current actions
    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.dino_duck:
            self.duck()
        if self.step_index >= 10:
            self.step_index = 0

        if not self.dino_duck and self.rect.y > self.Y_POS:
            self.rect.y = self.Y_POS
        if self.dino_duck and self.rect.y > self.Y_POS_DUCK:
            self.rect.y = self.Y_POS_DUCK

    # Handles jump physics
    def jump(self):
        self.image = JUMPING
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8

            if self.jump_vel < 0 and self.rect.y >= self.Y_POS:
                self.rect.y = self.Y_POS
                self.dino_jump = False
                self.dino_run = True
                self.jump_vel = self.JUMP_VEL

    # Manages running animation by alternating between sprite frames
    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1

    # Controls ducking animation and hitbox adjustments for collision detection
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        
        current_x = self.rect.x
        
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS 
        self.rect.y = self.Y_POS_DUCK
        self.step_index += 1

    # Provides a clean interface for the AI to control dinosaur states
    def set_state(self, run=False, jump=False, duck=False):
        self.dino_run = run
        self.dino_jump = jump
        self.dino_duck = duck

        if run:
            self.image = RUNNING[0]
        elif jump:
            self.image = JUMPING
        elif duck:
            self.image = DUCKING[0]

        if duck:
            self.rect = self.image.get_rect()
            self.rect.x = self.X_POS
            self.rect.y = self.Y_POS_DUCK
        else:
            if not jump:
                self.rect = self.image.get_rect()
                self.rect.x = self.X_POS
                self.rect.y = self.Y_POS

        if not jump:
            self.jump_vel = self.JUMP_VEL

    # Renders the dinosaur and debug information like collision lines to obstacles
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x,
                         self.rect.y, self.rect.width, self.rect.height), 2)
        for obstacle in obstacles:
            pygame.draw.line(SCREEN, self.color, (self.rect.x +
                             54, self.rect.y + 12), obstacle.rect.center, 2)


# Creates decorative cloud elements to enhance the game's aesthetics
class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    # Moves clouds at game speed and recycles them when off-screen
    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    # Renders clouds to the game screen
    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


# Base class for all game obstacles to standardize behavior
class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    # Moves obstacles toward the player and removes them when off-screen
    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    # Renders obstacles to the game screen
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


# Specialized obstacle class for small cacti with appropriate height placement
class SmallCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 325


# Specialized obstacle class for large cacti with appropriate height placement
class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300


# Flying obstacle with variable heights to challenge the AI in different ways
class Bird(Obstacle):
    BIRD_HEIGHT_HIGH = 250
    BIRD_HEIGHT_LOW = 310

    def __init__(self, image):
        self.type = 0
        self.image = image
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

        self.height = random.randint(0, 1)
        if self.height == 0:
            self.rect.y = self.BIRD_HEIGHT_HIGH
        else:
            self.rect.y = self.BIRD_HEIGHT_LOW

        self.index = 0

    # Updates bird position and animates wing flapping
    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

        self.index += 1
        if self.index >= 10:
            self.index = 0

    # Renders bird with animated wings
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.index // 5], self.rect)


# Removes a dinosaur from the game when it collides with an obstacle and records its performance data
def remove(index):
    if points > 0:
        current_gen_scores.append(points)
        current_gen_fitnesses.append(ge[index].fitness)

    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)


# Calculates Euclidean distance between two points for AI decision-making
def distance(pos_a, pos_b):
    dx = pos_a[0]-pos_b[0]
    dy = pos_a[1]-pos_b[1]
    return math.sqrt(dx**2+dy**2)


# Creates visual performance graphs to track AI learning progress across generations
def create_graphs():
    if len(generation_stats['generation']) > 0:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)

        axs[0].plot(generation_stats['generation'],
                    generation_stats['max_fitness'], 'b-', label='Max Fitness')
        axs[0].plot(generation_stats['generation'],
                    generation_stats['avg_fitness'], 'g--', label='Avg Fitness')
        axs[0].set_title('Fitness over Generations')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Fitness')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(generation_stats['generation'],
                    generation_stats['max_score'], 'r-', label='Max Score')
        axs[1].plot(generation_stats['generation'],
                    generation_stats['avg_score'], 'y--', label='Avg Score')
        axs[1].set_title('Score over Generations')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Score')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        fig.savefig('neat_progress.png')
        plt.close(fig)

        graph_surface = pygame.image.load('neat_progress.png')
        
        graph_surface = pygame.transform.scale(graph_surface, (500, 250))
        return graph_surface

    return None


# Core NEAT evaluation function that tests each genome's performance in the game environment
def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, obstacles, dinosaurs, ge, nets, points, current_gen_scores, current_gen_fitnesses

    current_gen_scores = []
    current_gen_fitnesses = []

    clock = pygame.time.Clock()
    points = 0
    cloud = Cloud()

    obstacles = []
    dinosaurs = []
    ge = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    MAX_DINOSAURS = 15
    genomes = genomes[:MAX_DINOSAURS]

    for idx, (genome_id, genome) in enumerate(genomes):
        dinosaurs.append(Dinosaur(dino_index=idx))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    # Increases score over time and accelerates game difficulty by increasing speed
    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f'Points:  {str(points)}', True, (0, 0, 0))
        SCREEN.blit(text, (950, 50))

    # Displays real-time information about the current generation's progress
    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(
            f'Dinosaurs Alive:  {str(len(dinosaurs))}', True, (0, 0, 0))
        text_2 = FONT.render(
            f'Generation:  {pop.generation+1}', True, (0, 0, 0))
        text_3 = FONT.render(
            f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_2, (50, 480))
        SCREEN.blit(text_3, (50, 510))

        if len(generation_stats['max_fitness']) > 0:
            text_4 = FONT.render(
                f'Best Fitness:  {max(generation_stats["max_fitness"]):.2f}', True, (0, 0, 0))
            text_5 = FONT.render(
                f'Best Score:  {max(generation_stats["max_score"])}', True, (0, 0, 0))
            SCREEN.blit(text_4, (50, 540))
            SCREEN.blit(text_5, (50, 570))

    # Creates a scrolling background effect to simulate movement
    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    graph_surface = create_graphs()

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        background()
        cloud.draw(SCREEN)
        cloud.update()

        if len(obstacles) == 0:
            rand_int = random.randint(0, 2)
            if rand_int == 0:
                obstacles.append(SmallCactus(
                    SMALL_CACTUS, random.randint(0, 2)))
            elif rand_int == 1:
                obstacles.append(LargeCactus(
                    LARGE_CACTUS, random.randint(0, 2)))
            elif rand_int == 2:
                obstacles.append(Bird(BIRD))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()

            i = 0
            while i < len(dinosaurs):
                if i < len(dinosaurs) and dinosaurs[i].rect.colliderect(obstacle.rect):
                    ge[i].fitness -= 1
                    remove(i)
                else:
                    i += 1

        i = 0
        while i < len(dinosaurs):
            if i >= len(dinosaurs):
                break

            dinosaur = dinosaurs[i]
            dinosaur.update()
            dinosaur.draw(SCREEN)

            if obstacles:
                obstacle = obstacles[0]

                if obstacle.rect.x + obstacle.rect.width < dinosaur.rect.x:
                    if not dinosaur.dino_jump:
                        dinosaur.set_state(run=True, jump=False, duck=False)
                else:
                    obstacle_distance = distance(
                        (dinosaur.rect.x, dinosaur.rect.y), obstacle.rect.midtop)
                    obstacle_height = obstacle.rect.y

                    is_bird = isinstance(obstacle, Bird)

                    output = nets[i].activate((
                        dinosaur.rect.y,
                        obstacle_distance,
                        obstacle_height,
                        1 if is_bird else 0
                    ))

                    ge[i].fitness += 0.1

                    if is_bird:
                        if obstacle.rect.y == Bird.BIRD_HEIGHT_HIGH and dinosaur.rect.y == dinosaur.Y_POS:
                            if output[0] > 0.5:
                                dinosaur.set_state(
                                    run=False, jump=False, duck=True)
                        elif output[0] > 0.5 and obstacle.rect.y == Bird.BIRD_HEIGHT_LOW:
                            dinosaur.set_state(
                                run=False, jump=True, duck=False)
                    else:
                        if output[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
                            dinosaur.set_state(
                                run=False, jump=True, duck=False)

            i += 1

        if len(dinosaurs) == 0:
            break

        statistics()
        score()

        if graph_surface:
            SCREEN.blit(graph_surface, (SCREEN_WIDTH -
                        520, SCREEN_HEIGHT - 270))

        pygame.display.update()
        clock.tick(30)

    if len(current_gen_fitnesses) > 0:
        generation_stats['max_fitness'].append(max(current_gen_fitnesses))
        generation_stats['avg_fitness'].append(
            sum(current_gen_fitnesses) / len(current_gen_fitnesses))
    else:
        generation_stats['max_fitness'].append(0)
        generation_stats['avg_fitness'].append(0)

    if len(current_gen_scores) > 0:
        generation_stats['max_score'].append(max(current_gen_scores))
        generation_stats['avg_score'].append(
            sum(current_gen_scores) / len(current_gen_scores))
    else:
        generation_stats['max_score'].append(0)
        generation_stats['avg_score'].append(0)

    generation_stats['species_count'].append(len(pop.species.species))
    generation_stats['generation'].append(pop.generation)

    save_stats()


# Saves performance metrics to a CSV file for external analysis
def save_stats():
    """Save the training statistics to a CSV file"""
    with open('neat_stats.csv', 'w') as f:
        f.write('Generation,MaxFitness,AvgFitness,MaxScore,AvgScore,SpeciesCount\n')

        for i in range(len(generation_stats['generation'])):
            f.write(f"{generation_stats['generation'][i]},{generation_stats['max_fitness'][i]:.2f},"
                    f"{generation_stats['avg_fitness'][i]:.2f},{generation_stats['max_score'][i]},"
                    f"{generation_stats['avg_score'][i]:.2f},{generation_stats['species_count'][i]}\n")


# Primary execution function that configures and runs the NEAT algorithm
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 50)

    with open('best_dino.pickle', 'wb') as f:
        import pickle
        pickle.dump(winner, f)

    create_graphs()
    print("Training complete! Check the stats and graphs in the current directory.")


if __name__ == '__main__':
    plt.switch_backend('agg')

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)