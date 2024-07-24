# Prado-Museum-Pictures-PageRank-Ranking-System
This project implements a ranking system for images using the PageRank algorithm. The dataset used is the "Prado Museum Pictures" dataset from Kaggle. Images are linked if they share at least one common tag, and the PageRank algorithm is used to rank these images based on their link structure.

Certainly! Here's a comprehensive GitHub report for your PageRank project. This report covers the project overview, implementation details, optimization techniques, and usage instructions.

---

# Prado Museum Pictures PageRank Ranking System

## Project Overview

This project implements a ranking system for images using the PageRank algorithm. The dataset used is the "Prado Museum Pictures" dataset from Kaggle. Images are linked if they share at least one common tag, and the PageRank algorithm is used to rank these images based on their link structure.

## Key Features

1. **Data Loading**: Efficient loading of data from a CSV file.
2. **Data Preprocessing**: Handles missing values, removes duplicates, and processes tags.
3. **Graph Construction**: Builds an undirected or weighted graph based on image tags.
4. **PageRank Computation**: Calculates the PageRank of images using NetworkX.
5. **Comparison of Strategies**: Compares results from different graph-building strategies.
6. **Validation and Visualization**: Validates results and visualizes the graph.

## Implementation Details

### Class `PradoPageRank`

The `PradoPageRank` class encapsulates the entire workflow for the PageRank system.

#### Methods:

1. **`__init__(self, file_path: str)`**
   - Initializes the class with the path to the dataset file.

2. **`load_data(self) -> None`**
   - Loads and explores the dataset with optimized settings.
   - **Optimizations**: Specifies data types, selects only necessary columns to save memory and speed up loading.

3. **`preprocess_data(self) -> None`**
   - Preprocesses the data for graph construction.
   - **Optimizations**: Utilizes efficient operations like `str.split` and `explode` to handle tags and `groupby` for aggregation.

4. **`build_graph(self, strategy: str = 'unweighted') -> None`**
   - Constructs the graph based on the chosen strategy (`'unweighted'` or `'weighted'`).
   - **Unweighted Strategy**: Adds edges between images that share tags.
   - **Weighted Strategy**: Adds edges with weights based on the number of shared tags.

5. **`compute_pagerank(self, weight: Optional[str] = None) -> None`**
   - Computes the PageRank of the images.
   - **Optimization**: Uses NetworkX’s efficient PageRank implementation.

6. **`compare_strategies(self) -> None`**
   - Compares the PageRank results obtained from different graph-building strategies.

7. **`validate_and_visualize(self) -> None`**
   - Validates the PageRank results and visualizes the graph using Matplotlib.

## Optimization Techniques

- **Data Loading**:
  - Specified data types to minimize memory usage.
  - Loaded only necessary columns to speed up the process.

- **Data Preprocessing**:
  - Utilized vectorized operations with pandas to efficiently process and aggregate tags.
  - Reduced complexity by using `groupby` and `explode` methods.

- **Graph Construction**:
  - Implemented efficient edge addition for both unweighted and weighted graphs.
  - Used dictionaries and default dictionaries for fast lookups and updates.

- **PageRank Computation**:
  - Leveraged NetworkX’s optimized PageRank algorithm to handle large graphs efficiently.

## Usage Instructions

1. **Install Dependencies**:
   Ensure you have the required packages installed. You can install them using pip:
   ```bash
   pip install pandas networkx matplotlib
   ```

2. **Prepare Dataset**:
   Place the `prado.csv` dataset in your working directory or update the file path in the script.

3. **Run the Script**:
   Execute the script to load data, preprocess it, build the graph, compute PageRank, and visualize the results:
   ```bash
   python prado_pagerank.py
   ```

4. **Review Results**:
   The script will output:
   - Data loading and preprocessing times.
   - Top images by PageRank for different graph-building strategies.
   - Visual representation of the image graph.

## Example Usage

```python
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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the content as needed to fit your specific project details and preferences.
