class try_class(object):

    def __init__(self):
        self.x = 4
        self.y = 40
        self._z = 50

    @property
    def myz(self):
        return self._z


init_try = try_class()
init_try