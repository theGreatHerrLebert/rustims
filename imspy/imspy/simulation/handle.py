import pandas as pd
import numpy as np

import imspy_connector as ims


class TimsTofSyntheticsDataHandleRust:
    def __init__(self, path: str):
        self.path = path
        self.__handle = ims.PyTimsTofSyntheticsDataHandle(path)

    def get_py_ptr(self):
        return self.__handle

    def __repr__(self):
        return f"TimsTofSyntheticsDataHandleRust(path={self.path})"
