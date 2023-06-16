import threading


class ConcurrentBitmask:
    kBitWidth = 64
    """segment locking"""

    def __init__(self, n):
        self.size = (n + self.kBitWidth - 1) // self.kBitWidth
        self.bitmask = [0] * self.size
        self.mutexes = [threading.Lock() for _ in range(self.size)]

    def query_and_mark(self, x):
        one = 1
        with self.mutexes[x // self.kBitWidth]:
            if self.bitmask[x // self.kBitWidth] & (one << (x % self.kBitWidth)):
                return False
            else:
                self.bitmask[x // self.kBitWidth] |= one << (x % self.kBitWidth)
                return True
