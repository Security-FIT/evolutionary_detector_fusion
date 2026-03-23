from functools import wraps
import logging
import numpy as np
import time


logger = logging.getLogger(__name__)


"""Decorator to time a method and log the result."""


def time_method(func):
    times = []

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        times.append(total_time)
        logger.info(
            f"Function {func.__name__} took {total_time:.4f} seconds, running average of {sum(times)/len(times):.4f} (+- {np.std(times):.4f})."
        )
        return result

    return timeit_wrapper
