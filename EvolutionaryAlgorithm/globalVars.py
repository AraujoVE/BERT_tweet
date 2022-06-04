from typing import List, Type, Dict, Any

class GlobalVars:
    def __init__(self):
        self.data : Dict[str,Any] = {} 
        pass
    
    def setAttr(self, name : str, value : Any) -> None :
        self.data[name] = value