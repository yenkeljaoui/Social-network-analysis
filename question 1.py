import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp

# Step 1: Load the social network graph from SNAP dataset
G = nx.read_edgelist('facebook_combined.txt.gz', create_using=nx.Graph(), nodetype=int)

# Step 2: Find the node with the highest degree, call it 'Adam'
adam_node = max(G.degree, key=lambda x: x[1])[0]

# Step 3: Create Adam's neighborhood graph (ego network)
adam_neighbors = list(G.neighbors(adam_node))
adam_subgraph = G.subgraph([adam_node] + adam_neighbors)

# Step 4: Compute requested statistics:

# Number of nodes in Adam's neighborhood graph
num_nodes = adam_subgraph.number_of_nodes()

# Number of edges in Adam's neighborhood graph
num_edges = adam_subgraph.number_of_edges()

# Number of connected components in Adam's neighborhood graph
connected_components = list(nx.connected_components(adam_subgraph))
num_connected_components = len(connected_components)

# Size of the largest connected component
largest_component_size = max(len(c) for c in connected_components)

# Degree distribution within Adam's neighborhood graph
degree_distribution = [adam_subgraph.degree(n) for n in adam_subgraph.nodes]

# Number of triangles involving Adam
num_triangles_involving_adam = sum(nx.triangles(adam_subgraph, nodes=[adam_node]).values())

# Maximum possible number of triangles in Adam's neighborhood graph (including Adam)
max_triangles = num_nodes * (num_nodes - 1) * (num_nodes - 2) // 6

# Adam's best friend: node with the highest number of mutual friends with Adam
adam_best_friend = max(adam_neighbors, key=lambda x: len(set(G.neighbors(x)) & set(adam_neighbors)))

# Step 5: Divide Adam's friends into subgroups using community detection algorithm
from networkx.algorithms import community
friend_subgroups = list(community.greedy_modularity_communities(adam_subgraph))

# Step 6: Display the computed statistics
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Number of connected components: {num_connected_components}")
#רכיב קשירות אצלנו יש אחד
print(f"Largest connected component size: {largest_component_size}")
#כמה קשתים יוצא מכל קודקוד כל מספר מייצג מישהו
print(f"Degree distribution: {degree_distribution}")
#משולש שלוש אנשים שחברים אחד של השני
print(f"Triangles involving Adam: {num_triangles_involving_adam}")
#כולם חברים של כולם בגרף
print(f"Maximum possible triangles: {max_triangles}")
print(f"Adam's best friend: {adam_best_friend}")

print("\nFriend subgroups among Adam's friends:")
for i, community in enumerate(friend_subgroups, start=1):
    print(f"Community {i}: {sorted(community)}")

# Step 7: Plot Adam's neighborhood graph with colored nodes
pos = nx.spring_layout(adam_subgraph)  # Generate positions for nodes

# Assign colors: Red for Adam, Green for Adam's best friend, Light Blue for others
node_colors = []
for node in adam_subgraph.nodes:
    if node == adam_node:
        node_colors.append('red')  # Adam's color
    elif node == adam_best_friend:
        node_colors.append('green')  # Adam's best friend color
    else:
        node_colors.append('lightblue')  # Other nodes

plt.figure(figsize=(10, 10))
nx.draw(adam_subgraph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
plt.title("Adam's Neighborhood Graph (Ego Network) with Colors")
plt.show()
