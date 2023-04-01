import os
from ast import literal_eval

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
    
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}
    
    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, "expecting each overide arg to be of form --arg=value, got %s" % arg
            key, val = keyval

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key[:2] == "--"
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = key[-1]

            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            print("command line overwriting config attribte %s with %s" % (key, val))
            setattr(obj, leaf_key, val)