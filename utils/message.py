from tkinter.messagebox import ABORT


EXIT = "exit"
REGISTER_MULTI = "register_multi"
REGISTER_SINGLE = "register_single"
ABORT = "abort"


class SignalMessage:
    def __init__(self, signal=EXIT, message={}):
        self.signal = signal
        self.message = message
