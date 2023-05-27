class Snake:

    def __init__(self):  # snek
        self.direction = 'RIGHT'
        self.position = [10, 5]
        self.length = 3
        self.body = [[10, 5], [9, 5], [8, 5], [7, 5]]
        self.alive = True
