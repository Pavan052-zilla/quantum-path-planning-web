import streamlit as st
import folium
from streamlit_folium import st_folium
import random

# === Import solvers (backend logic) ===
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np

# -------------------------------
# Backend: OR-Tools Baseline Solver
# -------------------------------
def solve_with_ortools(distance_matrix):
    """Simple TSP solver using OR-Tools"""
    num_nodes = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)  # 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route
    else:
        return None

# -------------------------------
# Placeholder Quantum Solver
# -------------------------------
def solve_with_quantum(distance_matrix):
    """
    Placeholder function.
    Later connect with D-Wave Ocean or Qiskit.
    For now, returns random permutation as 'quantum solution'.
    """
    nodes = list(range(len(distance_matrix)))
    route = [0] + random.sample(nodes[1:], len(nodes) - 1) + [0]
    return route

# -------------------------------
# Frontend (Streamlit UI)
# -------------------------------
st.set_page_config(page_title="Quantum Path Planning", layout="wide")

st.title("🚚 Quantum Path Planning (Demo)")
st.write("Compare **Classical (OR-Tools)** vs **Quantum (placeholder)** routing on a map.")

# User input: delivery points
st.sidebar.header("Configuration")
num_points = st.sidebar.slider("Number of delivery points", 3, 6, 4)

# Generate random coordinates (Amaravati bounding box approx.)
lat_min, lat_max = 16.48, 16.54
lon_min, lon_max = 80.5, 80.58
points = [(random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max)) for _ in range(num_points)]

# Distance matrix (Euclidean, for demo)
def haversine(p1, p2):
    import math
    R = 6371
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

distance_matrix = [[haversine(p1, p2) for p2 in points] for p1 in points]

# Solve with Classical + Quantum
classical_route = solve_with_ortools(distance_matrix)
quantum_route = solve_with_quantum(distance_matrix)

# -------------------------------
# Map Visualization
# -------------------------------
m = folium.Map(location=points[0], zoom_start=14)

# Plot points
for idx, (lat, lon) in enumerate(points):
    folium.Marker([lat, lon], popup=f"Point {idx}").add_to(m)

# Draw Classical Route
if classical_route:
    coords_classical = [points[i] for i in classical_route]
    folium.PolyLine(coords_classical, color="blue", weight=4, opacity=0.7, tooltip="Classical Route").add_to(m)

# Draw Quantum Route
if quantum_route:
    coords_quantum = [points[i] for i in quantum_route]
    folium.PolyLine(coords_quantum, color="red", weight=2, opacity=0.7, tooltip="Quantum Route").add_to(m)

st.subheader("Map of Routes")
st_map = st_folium(m, width=800, height=500)

# -------------------------------
# Results Comparison
# -------------------------------
st.subheader("📊 Route Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Classical (OR-Tools)**")
    st.write("Route:", classical_route)

with col2:
    st.markdown("**Quantum (Placeholder)**")
    st.write("Route:", quantum_route)
