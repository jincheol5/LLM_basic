import os

class DataUtils:
    base_path=os.path.join("..","data","llm","raw")

    @staticmethod
    def get_base_path():
        return DataUtils.base_path