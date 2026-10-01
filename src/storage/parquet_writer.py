import pandas as pd

class ParquetWriter:
    def __init__(self, output_path):
        self.output_path = output_path

    def write(self, data: list):
        df = pd.DataFrame(data)
        df.to_parquet(self.output_path)
