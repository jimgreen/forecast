"""Bounded fitted-model cache with full-input invalidation and per-key sharing."""
from collections import OrderedDict
from concurrent.futures import Future
from hashlib import sha256
from threading import Lock

import pandas as pd

from .pipeline import IntegratedForecastPipeline
from .schema import WEATHER_COLUMNS


class PipelineCache:
    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self.entries = OrderedDict()
        self.pending = {}
        self.lock = Lock()

    def get(self, history, targets, capacity_limits, scope):
        columns = list(dict.fromkeys((*WEATHER_COLUMNS, *targets)))
        # Backfills and corrections anywhere in history must invalidate the model.
        hashes = pd.util.hash_pandas_object(history[columns].sort_index(), index=True)
        key = (scope, sha256(hashes.to_numpy().tobytes()).digest(), tuple(targets),
               tuple(sorted(capacity_limits.items())))
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key]
            owner = key not in self.pending
            future = self.pending.setdefault(key, Future())
        if not owner:
            return future.result()
        try:
            # Different objects can fit concurrently; identical requests share work.
            pipeline = IntegratedForecastPipeline().fit(history, targets=targets, capacity_limits=capacity_limits)
            with self.lock:
                self.entries[key] = pipeline
                while len(self.entries) > self.maxsize:
                    self.entries.popitem(last=False)
            future.set_result(pipeline)
            return pipeline
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)
