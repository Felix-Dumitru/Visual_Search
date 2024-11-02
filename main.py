import time
import pygame
from queue import PriorityQueue

WIDTH = 800
HEIGHT = 800

WIN = pygame.display.set_mode(size = (WIDTH, HEIGHT))
pygame.display.set_caption("Path finding algorithms")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE= (255, 165, 0)
GREY = (128, 128, 128)

class Spot:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        #self.height = height
        self.x = row * width
        # self.y = col * height
        self.y = col * width
        self.color = WHITE
        self.neighbours = []
        self.total_rows = total_rows
        #self.total_cols = total_cols

    def get_pos(self):
        return self.row, self.col

    def is_visited(self):
        return self.color == RED

    #check if it's in the open set aka able to be visited next
    def is_open(self):
        return self.color == GREEN

    def is_wall(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == BLUE

    def reset(self):
        self.color = WHITE


    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_wall(self):
        self.color = BLACK

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = BLUE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall(): #DOWN
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():  #LEFT
                self.neighbours.append(grid[self.row][self.col - 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_wall(): #UP
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall():  #RIGHT
                self.neighbours.append(grid[self.row][self.col+1])

    def __lt__(self, other):
        return False

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return abs(x1- x2) + abs(y1 - y2)

# cols and height would be extra parameters if not square
def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

# cols and height would be extra parameters if not square
def clear_search(grid, rows, width):
    for row in grid:
        for spot in row:
            if not spot.is_start() and not spot.is_end() and not spot.is_wall():
                spot.reset()

def draw_grid(win, rows, width):
    gap = width // rows

    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def reconstruct_path(came_from, current, draw):

    current.make_end()

    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def print_path(came_from, start, end, draw):
    print("Path")
    print("Start")

    current = end
    reverse_path = []
    steps = 0

    while current in came_from:
        if current == end:
            pass
        x, y = current.get_pos()
        reverse_path.append((x,y))
        current = came_from[current]
        steps += 1

    x, y = start.get_pos()
    print(f" ({x}, {y})")
    for i in range (len(reverse_path) - 1, 0, -1):
        print(f" {reverse_path[i]}")
    x, y = end.get_pos()
    print(f" ({x}, {y})")

    print("Goal")
    print(f"Length: {steps} steps")

def dfs(draw, start, end):
    start_time = time.time()

    stack = [(start, [(None, start)])]
    explored = set()

    while stack:
        current, path  = stack.pop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if current in explored:
            continue
        explored.add(current)

        if current == end:
            end_time = time.time()
            total_time = end_time - start_time

            print("DFS")
            reconstruct_path({spot: prev for prev, spot in path[1:]}, end, draw)
            print_path({spot: prev for prev, spot in path[1:]}, start, end, draw)
            end.make_end()
            start.make_start()
            print(f"{total_time} seconds\n")
            return True

        for neighbour in current.neighbours:
            if neighbour not in explored:
                stack.append((neighbour, path + [(current, neighbour)]))
                neighbour.make_open()
                draw()

        if current != start:
            current.make_closed()

    return False

def bfs(draw, start, end):
    start_time = time.time()

    queue = [(start, [(None, start)])]
    explored = set()

    while queue:
        current, path = queue.pop(0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if current in explored:
            continue
        explored.add(current)

        if current == end:
            end_time = time.time()
            total_time = end_time - start_time

            print("BFS")
            reconstruct_path({spot: prev for prev, spot in path[1:]}, end, draw)
            print_path({spot: prev for prev, spot in path[1:]}, start, end, draw)
            start.make_start()
            end.make_end()
            print(f"{total_time} seconds\n")
            return True

        for neighbour in current.neighbours:
            if neighbour not in explored:
                queue.append((neighbour, path + [(current, neighbour)]))
                neighbour.make_open()
                draw()

        if current != start:
            current.make_closed()

    return False

def dijkstra(draw, grid, start, end):
    start_time = time.time()

    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    # to check if nodes are in the priority queue as I can't check the queue directly
    open_set_present = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_present.remove(current)

        if current == end:
            end_time = time.time()
            total_time = end_time - start_time

            reconstruct_path(came_from, end, draw)
            print("Dijkstra")
            print_path(came_from, start, end, draw)
            start.make_start()
            end.make_end()
            print(f"{total_time} seconds\n")

            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score

                if neighbour not in open_set_present:
                    count += 1
                    open_set.put((g_score[neighbour], count, neighbour))
                    open_set_present.add(neighbour)
                    neighbour.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def ucs(draw, start, end):
    start_time = time.time()

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_cost, current = open_set.get()

        if current == end:
            end_time = time.time()
            total_time = end_time - start_time

            print("UCS")
            reconstruct_path(came_from, end, draw)
            print_path(came_from, start, end, draw)
            start.make_start()
            end.make_end()
            print(f"{total_time} seconds\n")
            return True

        for neighbour in current.neighbours:
            new_cost = cost_so_far[current] + 1

            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current
                open_set.put((new_cost, neighbour))
                neighbour.make_open()
                draw()

        if current != start:
            current.make_closed()

    return False


def a_star(draw, grid, start, end):
    start_time = time.time()

    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    #filling everything with infinity using list comprehension
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    #to check if nodes are in the priority queue as I can't check the queue directly
    open_set_present = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_present.remove(current)

        if current == end:
            end_time = time.time()
            total_time = end_time - start_time

            reconstruct_path(came_from, end, draw)
            print("A*")
            print_path(came_from, start, end, draw)
            start.make_start()
            end.make_end()
            print(f"{total_time} seconds\n")

            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())

                if neighbour not in open_set_present:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_present.add(neighbour)
                    neighbour.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def main(win, width):
    rows = 50
    grid = make_grid(rows, width)

    start = None
    end = None

    run = True

    while run:
        draw(win, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: #LMB
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_wall()

            elif pygame.mouse.get_pressed()[2]: #RMB
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                spot = grid[row][col]

                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:

                #A*
                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)

                    a_star(lambda : draw(win, grid, rows, width), grid, start, end)

                #DFS
                if event.key == pygame.K_f and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)

                    dfs(lambda : draw(win, grid, rows, width), start, end)

                #BFS
                if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)

                    bfs(lambda: draw(win, grid, rows, width), start, end)

                #Dijkstra
                if event.key == pygame.K_d and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)

                    dijkstra(lambda: draw(win, grid, rows, width), grid, start, end)

                #UCS
                if event.key == pygame.K_u and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)

                    ucs(lambda: draw(win, grid, rows, width), start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(rows, width)

                if event.key == pygame.K_x:
                    clear_search(grid, rows, width)

    pygame.quit()

main(WIN, WIDTH)