from os import stat
import sys
import pandas as pd

from collections import namedtuple
from datetime import datetime, timedelta
import multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple


class Timer:
    lock_list = []
    records_list: List[List[Tuple]] = []
    _is_enable_list: List[bool] = None
    base_time_list: List[float] = None
    world_size = None

    @staticmethod
    def init(world_size):
        print(f'using {world_size=}')
        Timer.world_size = world_size
        Timer.lock_list = [mp.Lock() for _ in range(world_size)]
        Timer.records_list = [[] for _ in range(world_size)]
        Timer._is_enable_list = [False for _ in range(world_size)]
        Timer.base_time_list = [0.0 for _ in range(world_size)]

    def __init__(self, rank, label=None):
        self.rank = rank
        if Timer._is_enable_list[self.rank]:
            self.start = None
            self.label = label
            self.frame = sys._getframe(1)
            self.position = (self.frame.f_code.co_name, self.frame.f_lineno, self.label)

    def __enter__(self):
        now = datetime.now().timestamp()
        if Timer._is_enable_list[self.rank]:
            self.start = now - Timer.base_time_list[self.rank]

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = datetime.now().timestamp()
        if Timer._is_enable_list[self.rank]:
            end = now - Timer.base_time_list[self.rank]
            with Timer.lock_list[self.rank]:
                Timer.records_list[self.rank].append([*self.position, self.start, end])

    @staticmethod
    def stats(rank):
        return pd.DataFrame(Timer.records_list[rank], 
                            columns=['method', 'lineno', 'label', 'start', 'end'])

    @staticmethod
    def set_base_time(rank, base_time):
        with Timer.lock_list[rank]:
            Timer.base_time_list[rank] = base_time

    @staticmethod
    def enable(rank):
        Timer._is_enable_list[rank] = True

    @staticmethod
    def disable(rank):
        Timer._is_enable_list[rank] = False

    @property
    def is_enable(rank):
        return Timer._is_enable_list[rank]