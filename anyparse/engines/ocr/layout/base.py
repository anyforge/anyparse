from abc import ABC, abstractmethod

class BaseLayoutModel(ABC):
    """Base class for layout detectors.

    Defines a unified interface for layout detection.
    """
    @abstractmethod
    def start(self):
        """Start the detector (e.g., start worker processes)."""

    @abstractmethod
    def stop(self):
        """Stop the detector (e.g., stop worker processes)."""
        
    @abstractmethod
    def invoke(self):
        """Invoke the detector on a batch of images."""
        
    @abstractmethod
    async def ainvoke(self):
        """Ainvoke the detector on a batch of images."""