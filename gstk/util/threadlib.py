import sys
import threading
import traceback


def print_thread_stacks():
    current_thread = threading.current_thread()
    for thread in threading.enumerate():
        print(f"Thread ID: {thread.ident}, Name: {thread.name}")
        if thread is current_thread:
            # Print current thread stack.
            traceback.print_stack()
        else:
            # Print the stacks of all non-main threads.
            stack = sys._current_frames().get(thread.ident)
            if stack:
                traceback.print_stack(stack)
