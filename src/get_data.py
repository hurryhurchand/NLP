import pandas as pd
class DataFetcher():
    def __init__(self,path):
        self.path = path
    
    def read_file(self):
        df = pd.read_csv(self.path)
        print(df.head(3))
        return