import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

# Set plot style
plt.style.use('ggplot')
sns.set(style='whitegrid')

def visualize_product_distribution(data: pd.DataFrame) -> plt.Figure:
    """
    Visualize the distribution of products in the complaint data.
    
    Args:
        data (pd.DataFrame): The complaint data
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Count products
    product_counts = data['product'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bar chart
    bars = product_counts.plot(kind='barh', ax=ax, color=sns.color_palette("viridis", len(product_counts)))
    
    # Add count labels to bars
    for i, (count, bar) in enumerate(zip(product_counts, bars.patches)):
        ax.text(count + (count * 0.01), bar.get_y() + bar.get_height()/2, 
                f"{count:,}", va='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Number of Complaints')
    ax.set_ylabel('Product')
    ax.set_title('Distribution of Complaints by Product')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_issue_distribution(data: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Visualize the distribution of top issues in the complaint data.
    
    Args:
        data (pd.DataFrame): The complaint data
        top_n (int): Number of top issues to display
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Count issues and get top N
    issue_counts = data['issue'].value_counts().head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bar chart
    bars = issue_counts.plot(kind='barh', ax=ax, color=sns.color_palette("mako", len(issue_counts)))
    
    # Add count labels to bars
    for i, (count, bar) in enumerate(zip(issue_counts, bars.patches)):
        ax.text(count + (count * 0.01), bar.get_y() + bar.get_height()/2, 
                f"{count:,}", va='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Number of Complaints')
    ax.set_ylabel('Issue')
    ax.set_title(f'Top {top_n} Issues in Complaints')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_product_issue_heatmap(data: pd.DataFrame, top_products: int = 5, top_issues: int = 5) -> plt.Figure:
    """
    Create a heatmap showing the relationship between top products and top issues.
    
    Args:
        data (pd.DataFrame): The complaint data
        top_products (int): Number of top products to include
        top_issues (int): Number of top issues to include
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Get top products and issues
    top_product_values = data['product'].value_counts().head(top_products).index
    top_issue_values = data['issue'].value_counts().head(top_issues).index
    
    # Filter data to include only top products and issues
    filtered_data = data[data['product'].isin(top_product_values) & data['issue'].isin(top_issue_values)]
    
    # Create cross-tabulation
    crosstab = pd.crosstab(filtered_data['product'], filtered_data['issue'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    
    # Set title
    ax.set_title(f'Heatmap of Top {top_products} Products vs Top {top_issues} Issues')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_time_series(data: pd.DataFrame, date_column: str = 'date_received', 
                          freq: str = 'M', product: str = None) -> plt.Figure:
    """
    Visualize complaint trends over time.
    
    Args:
        data (pd.DataFrame): The complaint data
        date_column (str): The column containing date information
        freq (str): Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
        product (str): Filter for a specific product (optional)
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Ensure date column is datetime
    if data[date_column].dtype != 'datetime64[ns]':
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Filter by product if specified
    if product:
        filtered_data = data[data['product'] == product].copy()
    else:
        filtered_data = data.copy()
    
    # Set date as index
    filtered_data.set_index(date_column, inplace=True)
    
    # Count complaints by time period
    time_series = filtered_data.resample(freq).size()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    time_series.plot(ax=ax, marker='o', linestyle='-', color='#1f77b4')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Complaints')
    title = 'Complaint Trends Over Time'
    if product:
        title += f' for {product}'
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_narrative_length_distribution(data: pd.DataFrame, narrative_column: str = 'narrative') -> plt.Figure:
    """
    Visualize the distribution of narrative lengths.
    
    Args:
        data (pd.DataFrame): The complaint data
        narrative_column (str): The column containing narrative text
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Calculate narrative lengths
    narrative_lengths = data[narrative_column].str.len()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(narrative_lengths, bins=30, kde=True, ax=ax, color='#2ca02c')
    
    # Set labels and title
    ax.set_xlabel('Narrative Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Complaint Narrative Lengths')
    
    # Add vertical line for mean and median
    mean_length = narrative_lengths.mean()
    median_length = narrative_lengths.median()
    
    ax.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.0f}')
    ax.axvline(median_length, color='blue', linestyle='--', label=f'Median: {median_length:.0f}')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_retrieval_results(query: str, documents: List, scores: List = None) -> plt.Figure:
    """
    Visualize the relevance scores of retrieved documents.
    
    Args:
        query (str): The query
        documents (list): The retrieved documents
        scores (list): The relevance scores (optional)
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If scores are not provided, use placeholder values
    if not scores:
        scores = [1.0 - (0.1 * i) for i in range(len(documents))]
    
    # Prepare data for plotting
    doc_ids = [f"Doc {i+1}" for i in range(len(documents))]
    
    # Extract product and issue for labels
    labels = []
    for doc in documents:
        product = doc.metadata.get('product', 'Unknown')
        issue = doc.metadata.get('issue', 'Unknown')
        labels.append(f"{product} - {issue}")
    
    # Plot horizontal bar chart
    bars = ax.barh(doc_ids, scores, color=sns.color_palette("viridis", len(documents)))
    
    # Add labels to bars
    for i, (score, bar) in enumerate(zip(scores, bars)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{score:.4f}", va='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Relevance Score')
    ax.set_ylabel('Document')
    ax.set_title(f'Relevance Scores for Query: "{query}"')
    
    # Add document labels as a table
    table_data = []
    for i, label in enumerate(labels):
        table_data.append([f"Doc {i+1}", label])
    
    table = plt.table(cellText=table_data, 
                      colLabels=['ID', 'Document'],
                      loc='bottom',
                      cellLoc='left',
                      bbox=[0, -0.5, 1, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.3)
    
    return fig