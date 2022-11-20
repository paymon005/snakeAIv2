class Snake:

    def __init__(self):
        self._direction = 'RIGHT'
        self._position = [10, 5]
        self._length = 3
        self._body = [[10, 5], [9, 5], [8, 5], [7, 5]]
        self._alive = True

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value

    @direction.deleter
    def direction(self):
        del self._direction

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        self._body = value

    @body.deleter
    def body(self):
        del self._body

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @position.deleter
    def position(self):
        del self._position

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @length.deleter
    def length(self):
        del self._length

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, value):
        self._alive = value

    @alive.deleter
    def alive(self):
        del self._alive
