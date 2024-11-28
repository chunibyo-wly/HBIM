EXIT = "exit"
REGISTER_MULTI = "register_multi"
REGISTER_SINGLE = "register_single"
ABORT = "abort"
EXPORT = "export"


class SignalMessage:
    def __init__(self, signal=EXIT, message={}):
        self.signal = signal
        self.message = message
