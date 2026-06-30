from abc import ABC, abstractmethod


class AnyHealths(object):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def check_health(self, **kwargs):
        return True