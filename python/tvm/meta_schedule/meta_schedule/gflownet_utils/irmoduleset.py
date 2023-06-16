from ..module_equality import ModuleEquality


class Item:
    def __init__(self, mod, shash, module_equality="structural"):
        self.mod = mod
        self.shash = shash
        self.model_equality = ModuleEquality(module_equality)

    def __hash__(self):
        return self.shash

    def __eq__(self, other):
        return isinstance(other, Item) and self.shash == other.shash and self.model_equality.equal(self.mod, other.mod)

class IRModuleSet:
    def __init__(self,model_equality):
        self.tab = set()
        self.model_equality = model_equality

    def add(self, mod, shash):
        self.tab.add(Item(mod, shash,self.model_equality))

    def has(self, mod, shash):
        return Item(mod, shash,self.model_equality) in self.tab
    
    def Add(self,mod,shash):
        return self.add(mod,shash)
    
    def Has(self,mod,shash):
        return self.has(mod,shash)
