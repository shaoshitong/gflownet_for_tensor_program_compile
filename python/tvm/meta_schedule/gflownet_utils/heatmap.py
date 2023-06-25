import heapq
from dataclasses import dataclass, field
from typing import Any
from tvm.tir.schedule import Schedule,Trace


class Item:
    score: float
    sch: Schedule

    def __init__(self,score,sch):
        self.score=score
        self.sch=sch

    def __lt__(self, other):
        return self.score > other.score
    
class SizedHeap:
    def __init__(self, size_limit):
        self.size_limit = size_limit
        self.heap = []

    def push(self, score, sch):
        size = len(self.heap)
        item = Item(score, sch)
        if size < self.size_limit:
            # Heap is not full, just push
            heapq.heappush(self.heap, item)
        elif score > self.heap[0].score:
            # if the item is better than the worst one in the heap, we can safely kick it out
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, item)
        # Otherwise, the item is worse than any other element in the heap
    
    def __len__(self):
        return len(self.heap)
    
    def __getitem__(self,item):
        return self.heap[item]