import threading

class AbortableThread(threading.Thread):
    def __init__(self, target_function, *args, **kwargs):
        super().__init__()
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs
        self._stop_event = threading.Event()

    def run(self):
        self.target_function(*self.args, **self.kwargs)

    def stop(self):
        self._stop_event.set()