import json
from typing import Any, Dict

from numpyencoder import NumpyEncoder


class JSONSaver:

    @staticmethod
    def dump(data: Dict[Any, Any], file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
