import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time


class PradoPageRank:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.graph = None
        self.pagerank_scores = None
        self.tag_to_pictures = defaultdict(set)

    def load_data(self) -> None:
        """Load and explore the dataset with optimized settings."""
        start_time = time.time()
        
        # Optimize loading by specifying data types and selecting only necessary columns
        data_types = {
            'picture_id': 'str',
            'work_tag': 'str'
        }
        
        # Load only the necessary columns
        self.df = pd.read_csv(self.file_path, usecols=['picture_id', 'work_tag'], dtype=data_types)
        
        # Drop duplicates and missing values
        self.df.dropna(subset=['picture_id', 'work_tag'], inplace=True)
        self.df.drop_duplicates(inplace=True)
        
        end_time = time.time()
        print(f"Data loaded in {end_time - start_time:.2f} seconds")
        
        print("Data Sample:")
        print(self.df.head())
        print("\nData Description:")
        print(self.df.describe())
        print("\nData Info:")
        print(self.df.info())

    def preprocess_data(self) -> None:
        """Preprocess the data for building the graph using efficient operations."""
        start_time = time.time()
        
        # Split the work_tag column and explode it into individual tags
        self.df['work_tag'] = self.df['work_tag'].str.split(';')
        exploded_df = self.df.explode('work_tag')
        
        # Group by tag and aggregate picture IDs
        tag_groups = exploded_df.groupby('work_tag')['picture_id'].apply(set).to_dict()
        
        # Update the tag_to_pictures dictionary
        self.tag_to_pictures.update(tag_groups)
        
        end_time = time.time()
        print(f"Data preprocessed in {end_time - start_time:.2f} seconds")

    def build_graph(self, strategy: str = 'unweighted') -> None:
        """Build a graph based on the specified strategy."""
        start_time = time.time()
        
        if strategy not in ['unweighted', 'weighted']:
            raise ValueError("Invalid strategy. Choose either 'unweighted' or 'weighted'.")

        self.graph = nx.Graph() if strategy == 'unweighted' else nx.DiGraph()
        
        if strategy == 'unweighted':
            # Add edges based on shared tags
            for pictures in self.tag_to_pictures.values():
                pictures_list = list(pictures)
                for i in range(len(pictures_list)):
                    for j in range(i + 1, len(pictures_list)):
                        self.graph.add_edge(pictures_list[i], pictures_list[j])
        
        elif strategy == 'weighted':
            tag_to_count = defaultdict(lambda: defaultdict(int))
            
            # Calculate weights based on the number of shared tags
            for _, row in self.df.iterrows():
                picture_id = row['picture_id']
                tags = row['work_tag']
                for tag in tags:
                    for other_picture in self.tag_to_pictures[tag]:
                        if other_picture != picture_id:
                            tag_to_count[picture_id][other_picture] += 1
            
            for pic1, neighbors in tag_to_count.items():
                for pic2, weight in neighbors.items():
                    self.graph.add_edge(pic1, pic2, weight=weight)
        
        end_time = time.time()
        print(f"Graph built in {end_time - start_time:.2f} seconds")

    def compute_pagerank(self, weight: Optional[str] = None) -> None:
        """Compute the PageRank of the graph."""
        start_time = time.time()
        
        if self.graph is None:
            raise ValueError("Graph not built. Please call build_graph() first.")
        
        self.pagerank_scores = nx.pagerank(self.graph, weight=weight)
        
        # Convert PageRank scores to DataFrame for easier analysis
        pagerank_df = pd.DataFrame(list(self.pagerank_scores.items()), columns=['picture_id', 'pagerank'])
        pagerank_df = pagerank_df.sort_values(by='pagerank', ascending=False)
        print("Top Pictures by PageRank:")
        print(pagerank_df.head())
        
        end_time = time.time()
        print(f"PageRank computed in {end_time - start_time:.2f} seconds")

    def compare_strategies(self) -> None:
        """Compare results from different graph-building strategies."""
        strategies = ['unweighted', 'weighted']
        results = {}
        
        for strategy in strategies:
            print(f"\nBuilding graph using {strategy} strategy...")
            self.build_graph(strategy=strategy)
            weight = 'weight' if strategy == 'weighted' else None
            self.compute_pagerank(weight=weight)
            results[strategy] = self.pagerank_scores
        
        # Compare top results
        for strategy, scores in results.items():
            pagerank_df = pd.DataFrame(list(scores.items()), columns=['picture_id', 'pagerank'])
            pagerank_df = pagerank_df.sort_values(by='pagerank', ascending=False)
            print(f"\nTop Pictures by PageRank ({strategy}):")
            print(pagerank_df.head())

    def validate_and_visualize(self) -> None:
        """Validate the results and visualize the graph."""
        if self.graph is None:
            raise ValueError("Graph not built. Please call build_graph() first.")
        
        # Validate results
        if self.pagerank_scores is not None:
            top_pictures = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\nValidated Top Pictures by PageRank:")
            for pic, rank in top_pictures:
                print(f"Picture ID: {pic}, PageRank: {rank}")

        # Visualize the graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, k=0.5)
        nx.draw(self.graph, pos, with_labels=True, node_size=50, node_color='blue', font_size=8)
        plt.title("Prado Museum Pictures Graph")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Replace 'prado.csv' with the actual path to your dataset
    prado_pagerank = PradoPageRank(file_path='prado.csv')
    
    # Load and explore data
    prado_pagerank.load_data()
    
    # Preprocess data for building the graph
    prado_pagerank.preprocess_data()
    
    # Compare results from different strategies
    prado_pagerank.compare_strategies()
    
    # Validate and visualize
    prado_pagerank.validate_and_visualize()

