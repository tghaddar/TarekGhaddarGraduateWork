from sweep_solver import sort_priority

graph_indices = [0,3,4,7]
graphs_per_angle = 4

graph_indices = sort_priority(graph_indices,graphs_per_angle)
print(graph_indices)