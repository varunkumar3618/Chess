class EngineContainer(object):
    def __init__(self, engine):
        self.engine = engine

    def newGame(self):
        pass

    def position(self, position):
        self.engine.position(position)

    def go(self, moveTime):
        return self.engine.go(movetime=moveTime)

class UCIEngineContainer(EngineContainer):
    def newGame(self):
        self.engine.ucinewgame()
