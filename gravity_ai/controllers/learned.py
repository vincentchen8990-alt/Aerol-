"""Learned dynamics can be used for estimates, never bypassing the bounded controller."""
from gravity_ai.controllers.pid import TorquePID
class LearnedAssistedController(TorquePID):
    def __init__(self,model=None):
        super().__init__()
        self.model=model
    # Safety baseline remains the command issuer until independently verified.
