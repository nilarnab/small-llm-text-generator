import itertools
import random
from openpyxl import Workbook

def generate_graph_dataset(n, output_file="graph_sft_dataset.xlsx", variations_per_graph=5):
    """
    Generate a supervised fine-tuning dataset for graph encoding.
    Generates ALL possible graphs for each number of nodes from 1 to n.
    
    Args:
        n: Maximum number of nodes in graphs (will generate graphs from 1 to n nodes)
        output_file: Output Excel file name
        variations_per_graph: Number of different phrasings for each graph structure
    """
    
    # Connection phrases to use randomly
    connection_phrases = [
        "connected to",
        "joined to",
        "linked to",
        "attached to",
        "related to",
        "bound to",
        "tied to",
        "associated with"
    ]
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Graph Dataset"
    
    # Add headers
    ws.append(["QUESTION", "ANSWER"])
    
    dataset = []
    graph_counts = {}
    
    for num_nodes in range(1, n + 1):
        # Generate node names (A, B, C, D, ...)
        nodes = [chr(65 + i) for i in range(num_nodes)]
        
        if num_nodes == 1:
            # Skip single isolated node - we only want graphs with connections
            graph_counts[num_nodes] = 0
            continue
        else:
            # Generate all possible edges between nodes
            possible_edges = list(itertools.combinations(nodes, 2))
            
            # Generate ALL possible graphs with at least one edge
            # 2^|E| - 1 total graphs (excluding the empty graph)
            all_graphs = []
            
            # All non-empty subsets of edges (skip empty graph)
            for r in range(1, len(possible_edges) + 1):
                for edge_subset in itertools.combinations(possible_edges, r):
                    all_graphs.append(list(edge_subset))
            
            # Generate variations for each graph structure (all have edges now)
            for graph_edges in all_graphs:
                for variation in range(variations_per_graph):
                    input_text = generate_input_text(graph_edges, connection_phrases, variation)
                    output_text = generate_output_encoding(graph_edges, nodes)
                    dataset.append((input_text, output_text))
            
            graph_counts[num_nodes] = len(all_graphs) * variations_per_graph
    
    # Shuffle the dataset for better training
    random.shuffle(dataset)
    
    # Write to Excel
    for question, answer in dataset:
        print("Question:", question, "Answer:", answer)
        ws.append([question, answer])
    
    wb.save(output_file)
    
    print(f"Dataset generated with {len(dataset)} examples")
    print("\nBreakdown by number of nodes:")
    for num_nodes in range(2, n + 1):  # Start from 2 since we skip isolated nodes
        num_edges = num_nodes * (num_nodes - 1) // 2
        num_graphs = 2 ** num_edges - 1  # Subtract 1 to exclude empty graph
        print(f"  {num_nodes} nodes: {num_graphs} possible graphs (with edges) × {variations_per_graph} variations = {graph_counts[num_nodes]} examples")
    print(f"\nSaved to {output_file}")
    return len(dataset)


def generate_single_node_text(node, variation_seed):
    """Generate text for a single isolated node."""
    random.seed(variation_seed)
    
    templates = [
        f"Node {node} exists",
        f"{node} is a node",
        f"There is a node {node}",
        f"{node} stands alone",
        f"Single node {node}"
    ]
    
    return random.choice(templates) + "."


def generate_disconnected_nodes_text(nodes, variation_seed):
    """Generate text for disconnected nodes."""
    random.seed(variation_seed)
    
    node_list = " and ".join(nodes)
    
    templates = [
        f"Nodes {node_list} exist with no connections",
        f"{node_list} are disconnected nodes",
        f"There are nodes {node_list} with no edges",
        f"{node_list} are isolated nodes",
        f"Nodes {node_list} have no connections"
    ]
    
    return random.choice(templates) + "."


def generate_input_text(edges, phrases, variation_seed):
    """Generate natural language description of graph connections with variations."""
    random.seed(variation_seed)
    
    # Strategy 1: List connections individually
    if variation_seed % 3 == 0:
        sentences = []
        shuffled_edges = edges.copy()
        random.shuffle(shuffled_edges)
        for src, dst in shuffled_edges:
            phrase = random.choice(phrases)
            # Randomly swap order
            if random.random() > 0.5:
                sentences.append(f"{src} is {phrase} {dst}")
            else:
                sentences.append(f"{dst} is {phrase} {src}")
        return ". ".join(sentences) + "."
    
    # Strategy 2: Group by source node
    elif variation_seed % 3 == 1:
        edge_dict = {}
        for src, dst in edges:
            if src not in edge_dict:
                edge_dict[src] = []
            edge_dict[src].append(dst)
        
        sentences = []
        for node in sorted(edge_dict.keys()):
            connections = edge_dict[node]
            phrase = random.choice(phrases)
            if len(connections) == 1:
                sentences.append(f"{node} is {phrase} {connections[0]}")
            else:
                conn_str = " and ".join(connections)
                sentences.append(f"{node} is {phrase} {conn_str}")
        return ". ".join(sentences) + "."
    
    # Strategy 3: Mixed approach
    else:
        sentences = []
        remaining_edges = edges.copy()
        random.shuffle(remaining_edges)
        
        while remaining_edges:
            edge = remaining_edges.pop(0)
            src, dst = edge
            phrase = random.choice(phrases)
            
            if random.random() > 0.5:
                sentences.append(f"{src} is {phrase} {dst}")
            else:
                sentences.append(f"{dst} is {phrase} {src}")
        
        return ". ".join(sentences) + "."


def generate_output_encoding(edges, all_nodes):
    """Generate graph encoding from edges."""
    # Build adjacency information
    adjacency = {}
    
    for src, dst in edges:
        if src not in adjacency:
            adjacency[src] = []
        if dst not in adjacency:
            adjacency[dst] = []
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    
    # Sort for consistency
    for node in adjacency:
        adjacency[node] = sorted(set(adjacency[node]))
    
    # Generate encoding for nodes that have connections only
    encodings = []
    for node in sorted(all_nodes):
        if node in adjacency and adjacency[node]:
            sources = " ".join(adjacency[node])
            encodings.append(f"{sources} - {node}")
    
    return ", ".join(encodings)


# Example usage
if __name__ == "__main__":
    # Generate dataset with ALL possible graphs up to 3 nodes
    n = 3
    num_variations = 5
    total = generate_graph_dataset(n, "dataset/graph_sft_dataset.xlsx", variations_per_graph=num_variations)
    
    print(f"\nExpected totals:")
    print(f"2 nodes: 2^1 - 1 = 1 graph  (A-B)")
    print(f"3 nodes: 2^3 - 1 = 7 graphs (all combinations with at least one edge)")
    print(f"Total: (1 + 7) × {num_variations} = {8 * num_variations} examples")
    
    # Print some examples
    print("\n" + "="*80)
    print("Sample entries:")
    print("="*80)
    
    connection_phrases = ["connected to", "joined to", "linked to"]
    
    # Example 1: Simple 2-node graph
    edges = [("A", "B")]
    print(f"Input:  {generate_input_text(edges, connection_phrases, 0)}")
    print(f"Output: {generate_output_encoding(edges, ['A', 'B'])}")
    print()
    
    # Example 2: 3-node chain
    edges = [("A", "B"), ("B", "C")]
    print(f"Input:  {generate_input_text(edges, connection_phrases, 1)}")
    print(f"Output: {generate_output_encoding(edges, ['A', 'B', 'C'])}")
    print()
    
    # Example 3: 3-node triangle
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    print(f"Input:  {generate_input_text(edges, connection_phrases, 2)}")
    print(f"Output: {generate_output_encoding(edges, ['A', 'B', 'C'])}")