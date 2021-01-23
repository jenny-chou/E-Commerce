# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from zipfile import ZipFile


class Data:
    def __init__(self, zip_fpath, dest_path, data_fpath):
        self.data = None
        
        self._unzip_file(zip_fpath, dest_path)
        self._load_data(data_fpath)
        self._drop_duplicates()
    
    def _unzip_file(self, zip_fpath, dest_path):
        """Unzip data in zip file to destination."""
        with ZipFile(zip_fpath, 'r') as zipObj:
            zipObj.extractall(dest_path)
            print("Unzipped dataset")
    
    def _load_data(self, data_fpath):
        """Read data CSV."""
        self.data = pd.read_csv(data_fpath, sep='\t')
        print("Loaded data")
        
    def _drop_duplicates(self):
        """Drop duplicates in dataset."""
        self.data.drop_duplicates(inplace=True)
        print("Cleaned data")
        
    
class Recommender:
    def __init__(self, data, num_recs, orders='order_number',
                 categories='l2', items='l3'):
        self.data = data.data
        self.N = num_recs        
        self.order_col = orders
        self.cat_col = categories
        self.item_col = items
        self.target = None
        self.recs = None
        self.scores = None
        
    def recommend(self, target):
        """Recommand items given the target product using 3 different methods."""
        self.target = target
        print(f"If user buy {self.target}:\n")
        
        self._popular_items()
        self._product_association()
        self._collaborative_filtering()
        
    def _popular_items(self):
        """Recommend most popular items in the category."""
        # Find the category the target item belongs to
        categories = set(self.data[self.data[self.item_col]==self.target][self.cat_col])
        for category in categories:
            # Find all items in the category
            group = self.data[self.data[self.cat_col]==category]
            
            # Recommend popular items in the category 
            self._find_recs(group.groupby(self.item_col).size())
            self._print_recs("popular items in the category")
            
    def _product_association(self):
        """Recommend most popular items other customers also bought."""
        # Find orders that bought the target item
        orders = self.data[self.data[self.item_col]==self.target][self.order_col]
        
        # Find all items in those orders
        items = self.data[self.data[self.order_col].isin(orders)]
        
        # Recommend items frequently bought with the target item        
        self._find_recs(items.groupby(self.item_col).size())
        self._print_recs("popular items other customers bought")
        
    def _collaborative_filtering(self):
        """Recommend most popular items bought together."""
        jaccard_scores = pd.Series(0.0, index=self.data.l3.unique())
        
        # Group order numbers by item
        orders = self.data.groupby(self.item_col)[self.order_col]
        
        # Find order numbers that bought the target item 
        target_order_nums = set(orders.get_group(self.target))
        
        for i in jaccard_scores.index.tolist():
            # Find order numbers that bought the sample item 
            sample_order_nums = set(orders.get_group(i))
            
            # Calculate Jaccard similarity between target and sample item 
            # based on overlapping of order numbers
            intersection = len(target_order_nums.intersection(sample_order_nums))
            union = len(target_order_nums.union(sample_order_nums))
            jaccard_scores[i] = intersection/union
        
        # Recommend items that has most overlap with target item
        self._find_recs(jaccard_scores, True)
        self._print_recs("similar products", True)
       
    def _find_recs(self, df, with_scores=False):
        """Find top recommendations from sorted list."""
        df.drop(self.target, axis=0, inplace=True)
        
        # Sort items by popularity
        self.recs = list(df.sort_values(ascending=False).index)
        # Recommend only top N items
        self.recs = self.recs[:self.N] if len(self.recs)>=self.N else self.recs        
       
        if with_scores:
            self.scores = list(df.sort_values(ascending=False).values)
            self.scores = self.scores[:self.N] if len(self.scores)>=self.N else self.scores
        
    def _print_recs(self, statement, with_scores=False):
        """Print recommendations and scores."""
        print("Top", str(self.N), statement + ":\n", self.recs)        
        if with_scores:
            print("With scores:", ["%.5f" % score for score in self.scores])
        print()
                    

def main():
    # Process data
    zip_fpath = "e-corp-data.zip"
    dest_path = "data"
    data_fpath = os.path.join(dest_path, 'All Transations - 2 Weeks.txt')
    data = Data(zip_fpath, dest_path, data_fpath)
    
    # Itinialize recommendation engine
    num_recs = 5
    rec_engine = Recommender(data, num_recs)
    
    # Pick a target item and recommend other items
    target = np.random.choice(list(set(data.data.l3)))    
    rec_engine.recommend(target)
      
        
if __name__ == "__main__":
    main()