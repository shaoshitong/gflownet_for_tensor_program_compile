class Item:
    def __init__(self, mod, shash):
        self.mod = mod
        self.shash = shash

    def __hash__(self):
        return self.shash

    def __eq__(self, other):
        return isinstance(other, Item) and self.shash == other.shash and ModuleEquality.equal(self.mod, other.mod)

class IRModuleSet:
    def __init__(self):
        self.tab = set()

    def add(self, mod, shash):
        self.tab.add(Item(mod, shash))

    def has(self, mod, shash):
        return Item(mod, shash) in self.tab
