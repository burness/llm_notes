import os

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _str__(self):
        return self._str_helper(0)
    
    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" %k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%sL %s\n" % (k, v))
        parts = [" " *(indent * 4) + p for p in parts]
        return "".join(parts)
    
    