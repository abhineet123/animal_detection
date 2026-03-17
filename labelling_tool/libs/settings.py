import pickle
import os
import sys

class Settings(object):
    def __init__(self, path=None):
        if path is None:
            # Be default, the home will be in the same folder as labelImg
            home = os.path.expanduser(".")
            self.path = os.path.join(home, '.labelImgSettings.pkl')
        else:
            self.path = path
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
            return True
        return False

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.data = pickle.load(f)
                return True
        return False

