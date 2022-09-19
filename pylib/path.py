import os

join = os.path.join

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)