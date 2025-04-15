import numpy as np

from src.others.frame import Frame
from src.others.local_map import Map
from config import SETTINGS


THETA_MIN = SETTINGS["convisibility"]["min_common_edges"]
ESSENTIAL_THETA_MIN = SETTINGS["convisibility"]["essential_common_edges"]


"""
Convisibility Graph: Contains 2 things:
                     1. All keyframe poses
                     2. Edges (pose connections) with weight >= 15
                        (weight == common landmark observations)
Spanning Tree: Minimalistic Convisibility Graph. Contains 2 things:
               1. All keyframe poses
               2. Edges only with the previous keyframe that shares the most observations
Essential Graph: Constains 3 things:
                 1. The Spanning Tree
                 2. Convisibility Edges with weight >= 100
                 3. Loop Closure edges
"""

class Graph:
    def __init__(self):
        self.nodes = {} # Dictionary keyframe_id -> observed map_point_ids
        self.edges = {} # Dictionary (kf_id1, kf_id2) -> weight (common map points)

    def _add_node(self, kf_id: int, mp_ids: set):
        self.nodes[kf_id] = mp_ids

    def _add_edge(self, kf_id1: int, kf_id2: int, weight: int):
        edge_id = tuple(sorted((kf_id1, kf_id2)))
        self.edges[edge_id] = weight

    def _remove_node(self, kf_id: int):
        del self.nodes[kf_id]

    def _remove_edges_with(self, kf_id: int):
        edges_to_remove = [edge_id for edge_id in self.edges.keys() if kf_id in edge_id]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]

    def _remove_edge(self, kf_id1: int, kf_id2: int):
        edge_id = tuple(sorted((kf_id1, kf_id2)))
        del self.edges[edge_id]

    def _reset(self):
        self.nodes = {}
        self.edges = {}

class ConvisibilityGraph(Graph):
    def __init__(self):
        """
        Initialize the Covisibility Graph.
        
        Args:
            min_common_points (int): Minimum number of common map points to form an edge in the covisibility graph.
            essential_theta_min (int): Threshold for covisibility edge weight to be included in the Essential Graph.
        """
        # Spanning tree edges: built incrementally when keyframes are added.
        self.spanning_tree = Graph()
        # Essential Graph
        self.essential_graph = Graph()
        # Loop closure edges added from separate loop detection.
        self.loop_closure_edges = {}

    def add_keyframe(self, keyframe: Frame, map: Map):
        """
        Adds a new keyframe to the graph and updates the covisibility edges and the spanning tree.

        Args:
            keyframe: Unique identifier for the keyframe.
            map_points (iterable): Collection of map point identifiers observed in the keyframe.
        """
        if keyframe.id in self.nodes.keys():
            print(f"Keyframe {keyframe.id} already exists!")
            return
        
        # Store the new keyframe observations (convert to set for easy intersection)
        kf_map_pt_ids = map.get_frustum_point_ids(keyframe)
        if len(kf_map_pt_ids) == 0:
            print(f"Keyframe {keyframe.id} observes 0 map points!")
            return
        self._add_node(keyframe.id, kf_map_pt_ids)
        self.spanning_tree._add_node(keyframe.id, kf_map_pt_ids)
        
        # Spanning tree connections
        best_parent_id = None
        max_shared = -1
        
        # Loop over existing keyframes (if any) to update graph edges.
        for other_kf_id, other_kf_map_point_ids in self.nodes.items():
            if other_kf_id == keyframe.id:
                continue

            # Compute the number of shared map points with the current keyframe.
            num_shared_points = np.intersect1d(other_kf_map_point_ids, kf_map_pt_ids).size()
            
            # For the covisibility graph, we only add an edge if num_shared_points observations >= threshold.
            if num_shared_points >= THETA_MIN:
                edge_id = tuple(sorted((other_kf_id, keyframe.id)))
                # If the edge already exists, we update it with the maximum num_shared_points count observed.
                if edge_id in self.edges.keys():
                    weight = max(self.edges[edge_id], num_shared_points)
                else:
                    weight = num_shared_points
                self._add_edge(other_kf_id, keyframe.id, weight)

                # If the weight is >= 100, add to the Essential Graph
                if weight >= ESSENTIAL_THETA_MIN:
                    self.essential_graph._add_edge(other_kf_id, keyframe.id, weight)
            
            # For the spanning tree, choose the keyframe that shares the most map points.
            if num_shared_points > max_shared:
                best_parent_id = other_kf_id
                max_shared = num_shared_points
        
        # If there is a valid parent keyframe, add the connection in the spanning tree.
        if best_parent_id is not None:
            self.spanning_tree._add_edge(best_parent_id, keyframe.id, max_shared)
            # All the spanning tree edges go to the Essential Graph too
            self.essential_graph._add_edge(best_parent_id, keyframe.id, max_shared)

    def get_frustum_point_ids(self, kf_id: int):
        """Returns the map points seen by a keyframe"""
        return self.nodes[kf_id]
    
    def get_frustum_points(self, t_frame: Frame, map: Map):
        """Returns the points that are in the view of a given frame"""
        points = set()
        for p_id in self.nodes[t_frame.id]:
            points.add[map.points[p_id]]
        return points
    
    def get_connected_frames_and_their_points(self, kf_id: int) -> tuple[set[int], set[int]]:
        """Returns the keyframes connected to a specific keyframe and the map points seen by them."""
        # Keep the connected nodes and points
        connected_kf_ids = set()
        connected_kf_point_ids = set()
        # Iterate over all the edges
        for (kf1_id, kf2_id) in self.edges.keys():
            # Check if this node is part of this edge
            # If it is, the other node and its points are of interest
            if kf1_id == kf_id:
                connected_kf_ids.add(kf2_id)
                connected_kf_point_ids.add(self.nodes[kf2_id])
            elif kf2_id == kf_id:
                connected_kf_ids.add(kf1_id)
                connected_kf_point_ids.add(self.nodes[kf1_id])

        return connected_kf_ids, connected_kf_point_ids
    
    def get_reference_frame(self, kf_id: int) -> int:
        """Returns the keyframe connected to a given keyframe that shares the most map points"""
        ref_frame_id = -1
        max_weight = -1
        # Iterate over all the edges
        for (kf1_id, kf2_id), weight in self.edges.items():
            # Check if this node is part of this edge
            if kf1_id == kf_id:
                if weight > max_weight:
                    max_weight = weight
                    ref_frame_id = kf1_id
            elif kf2_id == kf_id:
                if weight > max_weight:
                    max_weight = weight
                    ref_frame_id = kf2_id

        return ref_frame_id
    
    def get_neighbor_frames_and_their_points(self, kf_ids: set) -> tuple[set[int], set[int]]:
        """Returns all the neighboring keyframes"""
        neighbors = set()
        neighbor_points = set()
        # Iterate over all the edges
        for (kf1_id, kf2_id) in self.edges.keys():
            # Search for a neighbor
            if kf1_id in kf_ids and kf2_id not in kf_ids:
                neighbors.add(kf2_id)
                neighbor_points.add(self.nodes[kf2_id])
            elif kf1_id not in kf_ids and kf2_id in kf_ids:
                neighbors.add(kf1_id)
                neighbor_points.add(self.nodes[kf1_id])

        return neighbors, neighbor_points
    
    def remove_keyframe(self, keyframe: Frame):
        """
        Removes a keyframe from the graph and updates any edges associated with it.

        Args:
            keyframe.id: The identifier of the keyframe to be removed.
        """
        if keyframe.id not in self.nodes:
            print(f"Keyframe {keyframe.id} does not exist.")
            return
        
        self._remove_node(keyframe.id)
        self._remove_edges_with(keyframe.id)
        
        self.spanning_tree._remove_node(keyframe.id)
        self.spanning_tree._remove_edges_with(keyframe.id)
        
        self.essential_graph._remove_node(keyframe.id)
        self.essential_graph._remove_edges_with(keyframe.id)
        
        # Remove loop closure edges that involve the removed keyframe.
        edges_to_remove = [edge for edge in self.loop_closure_edges if keyframe.id in edge]
        for edge in edges_to_remove:
            del self.loop_closure_edges[edge]
        
        # Note: In a fully featured system you might want to re-compute or update the spanning tree
        # to ensure full connectivity, but for simplicity we are only removing associated edges here.

    def create_local_map(self, frame: Frame, map: Map):
        """
        Projects the map into a given frame and returns a local map.
        This local map contains: 
        - The frames that share map points with the current frame -> K1
        - The neighboring frames of K1 in the convisibility graph -> K2
        - A reference frame Kref in K1 which shares the most points with the current frame
        """
        # Find the points of the current frame
        frame_point_ids = self.get_frustum_point_ids(frame.id)

        # Find the frames that share map points and their points
        K1_frame_ids, K1_point_ids = self.get_connected_frames_and_their_points(frame.id)

        # Find neighboring frames to K1 and their points
        K2_frame_ids, K2_point_ids = self.get_neighbor_frames_and_their_points(frame.id)

        # Find a reference frame
        ref_frame_id = self.get_reference_frame(frame.id)

        # Merge the point ids
        local_map_point_ids = K1_point_ids + K2_point_ids

        # Create a local map with the K1 and K2 points
        local_map = Map(ref_frame_id)
        for p_id in local_map_point_ids:
            local_map.points[p_id] = map.points[p_id]

        return local_map

    def add_loop_edge(self, keyframe_id1, keyframe_id2, weight):
        """
        Adds an edge corresponding to a loop closure.

        Args:
            keyframe_id1: First keyframe identifier.
            keyframe_id2: Second keyframe identifier.
            weight (int): The weight (e.g., number of common observations) for the loop closure edge.
        """
        if keyframe_id1 not in self.nodes or keyframe_id2 not in self.nodes:
            print("One or both keyframes do not exist.")
            return
        
        edge_id = tuple(sorted((keyframe_id1, keyframe_id2)))
        self.loop_closure_edges[edge_id] = weight

        # Add it to the essential graph
        self.essential_graph._add_edge(keyframe_id1, keyframe_id2, weight)

    def print_graphs(self):
        """
        Prints the current covisibility graph, spanning tree, loop closure edges, and the computed essential graph.
        """
        print("=== Covisibility Graph Edges ===")
        for edge, weight in self.edges.items():
            print(f"{edge}: {weight}")
        
        print("\n=== Spanning Tree Edges ===")
        for edge, weight in self.spanning_tree.edges.items():
            print(f"{edge}: {weight}")
        
        print("\n=== Loop Closure Edges ===")
        for edge, weight in self.loop_closure_edges.items():
            print(f"{edge}: {weight}")
        
        print("\n=== Essential Graph Edges ===")
        for edge, weight in self.essential_graph.edges.items():
            print(f"{edge}: {weight}")
