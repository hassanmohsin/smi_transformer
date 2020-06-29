from collections import OrderedDict
import json


def load_params(param_file):
    if param_file is not None:
        params = json.loads(open(param_file).read(),
                            object_pairs_hook=OrderedDict)
        return params
    else:
        return None
