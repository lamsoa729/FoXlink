# ==============================================================================
# PROFILING UTILITIES
# ==============================================================================

from functools import wraps
import time
import os

PROFILING_ENABLED = os.getenv("FOXLINK_PROFILE", "false").lower() == "true"


class ProfilerStats:
    """Simple profiler to track function call times."""

    def __init__(self):
        self.stats = {}
        self.call_counts = {}
        print(f"[DEBUG] ProfilerStats.__init__() - id: {id(self)}")

    def record(self, func_name, duration):
        # print(f"[DEBUG] ProfilerStats.record() called - id: {id(self)}")
        # print(f"[DEBUG] Recording {func_name}: {duration:.6f}s")
        # print(f"[DEBUG] Current stats keys before: {list(self.stats.keys())}")

        if func_name not in self.stats:
            self.stats[func_name] = []
            self.call_counts[func_name] = 0
            print(f"[DEBUG] Created new entry for {func_name}")

        self.stats[func_name].append(duration)
        self.call_counts[func_name] += 1

        # print(f"[DEBUG] Current stats keys after: {list(self.stats.keys())}")
        # print(f"[DEBUG] {func_name} now has {self.call_counts[func_name]} calls")

    def print_stats(self):
        if not self.stats:
            print("=== No profiling data collected! ===")
            print(f"Stats dict is empty: {self.stats}")
            print(f"Call counts dict: {self.call_counts}")
            return

        print("\n=== Function Performance Stats ===")
        for func_name in sorted(self.stats.keys()):
            times = self.stats[func_name]
            count = self.call_counts[func_name]
            total_time = sum(times)
            avg_time = total_time / count
            max_time = max(times)
            min_time = min(times)

            print(
                f"{func_name:30} | "
                f"Calls: {count:6} | "
                f"Total: {total_time:8.4f}s | "
                f"Avg: {avg_time:8.6f}s | "
                f"Max: {max_time:8.6f}s | "
                f"Min: {min_time:8.6f}s"
            )


# Global profiler instance
profiler = ProfilerStats()


def get_current_profiler():
    """Always get the current global profiler instance."""
    global profiler
    return profiler


def profile_function(func):
    """Decorator to profile function execution time."""
    if not PROFILING_ENABLED:
        return func  # Return original function with zero overhead

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current profiler instance dynamically
        current_profiler = get_current_profiler()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time

        current_profiler.record(func.__name__, duration)

        if duration > 0.01:
            print(f"[PROFILE] {func.__name__}: {duration:.6f}s")

        return result

    return wrapper


def get_profiler_stats():
    """Get the global profiler stats."""
    return profiler


def print_profiler_stats():
    """Print all profiler stats."""
    profiler.print_stats()


def reset_profiler():
    """Reset profiler stats."""
    global profiler
    profiler = ProfilerStats()
