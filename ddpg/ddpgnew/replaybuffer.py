import random


class ReplayBuffer():
    def __init__(self, capacity):
        self._capacity = capacity
        self._list = []
        self._cursor = 0

    def add(self, item):
        if len(self._list) == self._capacity:
            self._list[self._cursor] = item
            self._cursor = (self._cursor + 1) % self._capacity
        else:
            self._list.append(item)

    def sample(self, n):
        return random.choices(self._list, k=n)
