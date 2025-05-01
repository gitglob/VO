from itertools import combinations
import src.globals as ctx
from config import SETTINGS, log


DEBUG = SETTINGS["generic"]["debug"]
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

    def _add_node(self, kf_id: int):
        self.nodes[kf_id] = set()

    def _add_node_with_points(self, kf_id: int, mp_ids: set):
        self.nodes[kf_id] = mp_ids

    def _add_edge(self, kf_id1: int, kf_id2: int, weight: int):
        edge_id = tuple(sorted((kf_id1, kf_id2)))
        self.edges[edge_id] = weight

    def _remove_node(self, kf_id: int):
        if kf_id in self.nodes.keys():
            del self.nodes[kf_id]

    def _remove_edges_with(self, kf_id: int):
        edges_to_remove = [edge_id for edge_id in self.edges.keys() if kf_id in edge_id]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]

    def _remove_edge(self, kf_id1: int, kf_id2: int):
        edge_id = tuple(sorted((kf_id1, kf_id2)))
        if edge_id in self.edges.keys():
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
        super().__init__()
        # Spanning tree edges: built incrementally when keyframes are added.
        self.spanning_tree = Graph()
        # Essential Graph
        self.essential_graph = Graph()
        # Loop closure edges added from separate loop detection.
        self.loop_closure_edges = {}

        self._first_keyframe_id = None

    def add_first_keyframe(self, kf_id: int):
        """Adds the first keyframe to the graph."""
        if DEBUG:
            log.info(f"[Graph] Buffering keyframe #{kf_id}")

        if kf_id in self.nodes.keys():
            log.warning(f"\t Keyframe {kf_id} already exists!")
            return
        
        self._first_keyframe_id = kf_id

    def add_keyframe(self, kf_id: int):
        """Adds a new keyframe to the graph.
        """
        if DEBUG:
            log.info(f"[Graph] Adding keyframe #{kf_id}!")

        if kf_id in self.nodes.keys():
            log.warning(f"\t Keyframe {kf_id} already exists!")
            return

        # Add the current keyframe
        self._add_node(kf_id)
        self.spanning_tree._add_node(kf_id)


    def add_init_keyframe(self, kf_id: int, kf_mp_ids: set[int]):
        """
        Adds a new keyframe to the graph and updates the covisibility edges and the spanning tree.

        Args:
            keyframe: Unique identifier for the keyframe.
            pairs: Dictionary matching a feature to a map point with a distance
        """
        if DEBUG:
            log.info(f"[Graph] Adding keyframe #{kf_id} with {len(kf_mp_ids)} map point observations!")

        if kf_id in self.nodes.keys():
            log.warning(f"\t Keyframe {kf_id} already exists!")
            return

        # Add the first keyframe if hanging
        if self._first_keyframe_id is not None:
            self._add_node_with_points(self._first_keyframe_id, kf_mp_ids)
            self.spanning_tree._add_node_with_points(self._first_keyframe_id, kf_mp_ids)
            self._first_keyframe_id = None

        # Add the current keyframe
        self._add_node_with_points(kf_id, kf_mp_ids)
        self.spanning_tree._add_node_with_points(kf_id, kf_mp_ids)

        self._update_edges_on_new_frame(kf_id, kf_mp_ids)

    def add_keyframe_with_points(self, kf_id: int, kf_mp_ids: set[int]):
        """Adds a new keyframe to the graph and updates the covisibility edges and the spanning tree."""
        if DEBUG:
            log.info(f"[Graph] Adding keyframe #{kf_id}")
        if kf_id in self.nodes.keys():
            log.warning(f"\t Keyframe {kf_id} already exists!")
            return
        
        # Store the new keyframe observations
        if DEBUG:
            log.info(f"\t Keyframe {kf_id} observes {len(kf_mp_ids)} map points!")
        self._add_node_with_points(kf_id, kf_mp_ids)
        self.spanning_tree._add_node_with_points(kf_id, kf_mp_ids)

        self._update_edges_on_new_frame(kf_id, kf_mp_ids)

    def add_observation(self, kf_id: int, pid: int):
        """Adds a point observation to a node"""
        self.nodes[kf_id].add(pid)

    def add_loop_edge(self, keyframe_id1: int, keyframe_id2: int, weight: int):
        """
        Adds an edge corresponding to a loop closure.

        Args:
            keyframe_id1: First keyframe identifier.
            keyframe_id2: Second keyframe identifier.
            weight (int): The weight (e.g., number of common observations) for the loop closure edge.
        """
        if keyframe_id1 not in self.nodes or keyframe_id2 not in self.nodes:
            log.warning("[Graph] One or both keyframes do not exist.")
            return
        
        edge_id = tuple(sorted((keyframe_id1, keyframe_id2)))
        self.loop_closure_edges[edge_id] = weight

        # Add it to the essential graph
        self.essential_graph._add_edge(keyframe_id1, keyframe_id2, weight)


    def _update_edges_on_new_frame(self, kf_id: int, map_pt_ids: set):
        """Updates the covisibility edges and the spanning tree given a new keyframe and its points."""
        # Spanning tree connections
        best_parent_id = None
        max_shared = -1
        
        # Loop over existing keyframes (if any) to update graph edges.
        num_new_edges = 0
        for other_kf_id, other_kf_map_point_ids in self.nodes.items():
            if other_kf_id == kf_id:
                continue

            # Compute the number of shared map points with the current keyframe.
            shared_points = other_kf_map_point_ids.intersection(map_pt_ids)
            num_shared_points = len(shared_points)
            
            # For the covisibility graph, we only add an edge if num_shared_points observations >= threshold.
            if num_shared_points >= THETA_MIN:
                weight = num_shared_points
                self._add_edge(other_kf_id, kf_id, weight)
                num_new_edges += 1

                # If the weight is >= 100, add to the Essential Graph
                if weight >= ESSENTIAL_THETA_MIN:
                    self.essential_graph._add_edge(other_kf_id, kf_id, weight)
            
            # For the spanning tree, choose the keyframe that shares the most map points.
            if num_shared_points > max_shared:
                best_parent_id = other_kf_id
                max_shared = num_shared_points
        
        # If there is a valid parent keyframe, add the connection in the spanning tree.
        if best_parent_id is not None:
            self.spanning_tree._add_edge(best_parent_id, kf_id, max_shared)
            # All the spanning tree edges go to the Essential Graph too
            self.essential_graph._add_edge(best_parent_id, kf_id, max_shared)

        if DEBUG:
            log.info(f"\t Added {num_new_edges} edges to keyframe {kf_id}!")

    def update_edges(self):
        """Updates the covisibility edges after points have been modified."""
        # Spanning tree connections
        best_parent_id = None
        max_shared = -1

        # Iterate over all node combinations
        for kf_id, other_kf_id in combinations(self.nodes.keys(), 2):
            kf_pt_ids = self.nodes[kf_id]
            other_kf_pt_ids = self.nodes[kf_id]
            # Skip node edges with itself
            assert(other_kf_id != kf_id)

            # Compute the number of shared map points with the current keyframe.
            shared_points = kf_pt_ids.intersection(other_kf_pt_ids)
            num_shared_points = len(shared_points)
        
            # For the covisibility graph, we only add an edge if num_shared_points observations >= threshold.
            if num_shared_points >= THETA_MIN:
                weight = num_shared_points
                self._add_edge(other_kf_id, kf_id, weight)

                # If the weight is >= 100, add to the Essential Graph
                if weight >= ESSENTIAL_THETA_MIN:
                    self.essential_graph._add_edge(other_kf_id, kf_id, weight)
        
            # For the spanning tree, choose the keyframe that shares the most map points.
            if num_shared_points > max_shared:
                best_parent_id = other_kf_id
                max_shared = num_shared_points
        
        # If there is a valid parent keyframe, add the connection in the spanning tree.
        if best_parent_id is not None:
            self.spanning_tree._add_edge(best_parent_id, kf_id, max_shared)
            # All the spanning tree edges go to the Essential Graph too
            self.essential_graph._add_edge(best_parent_id, kf_id, max_shared)

    
    def get_frame_point_ids(self, frame_id: int):
        """Returns the point ids that are in the view of a given frame"""
        points_ids = set()
        for pid in self.nodes[frame_id]:
            points_ids.add(pid)
        return points_ids    
    
    def get_frame_points(self, frame_id: int):
        """Returns the points that are in the view of a given frame"""
        points = set()
        for pid in self.nodes[frame_id]:
            points.add(ctx.map.points[pid])
        return points
        
    def get_connected_frames(self, kf_id: int, num_edges: int = None) -> set[int]:
        """Returns the keyframes connected to a specific keyframe.
        If num_edges is specified, only the num_edges strongest connections (highest weight) are returned.
        """
        # Gather all (other_frame_id, weight) pairs
        connections: list[tuple[int, float]] = []
        for (kf1_id, kf2_id), w in self.edges.items():
            if kf1_id == kf_id:
                connections.append((kf2_id, w))
            elif kf2_id == kf_id:
                connections.append((kf1_id, w))

        # If a limit was specified, sort by weight and trim
        if num_edges is not None:
            # sort descending by weight
            connections.sort(key=lambda pair: pair[1], reverse=True)
            connections = connections[:num_edges]

        # Return just the frame IDs (as a set)
        connection_frame_ids = {frame_id for frame_id, _ in connections}
    
        return connection_frame_ids

    def get_connected_frames_with_min_w(self, kf_id: int, min_weight: int) -> set[int]:
        """Returns the keyframes connected to a given keyframe that share the at least X map points"""
        connected_frame_ids = set()
        # Iterate over all the edges
        for (kf1_id, kf2_id), weight in self.edges.items():
            # Check if this node is part of this edge
            if kf1_id == kf_id:
                if weight > min_weight:
                    connected_frame_ids.add(kf2_id)
            elif kf2_id == kf_id:
                if weight > min_weight:
                    connected_frame_ids.add(kf1_id)

        return connected_frame_ids
 
    def get_frames_that_observe_points(self, pids: set[int]) -> set[int]:
        """Returns the keyframes that observe a set of points"""
        observing_kf_ids = set()
        for pid in pids:
            for kf_id, point_ids in self.nodes.items():
                if pid in point_ids:
                    observing_kf_ids.add(kf_id)

        return observing_kf_ids

    def get_connected_frames_and_their_points(self, kf_id: int) -> tuple[set[int], set[int]]:
        """Returns the keyframes connected to a specific keyframe and the map points seen by them."""
        # Keep the connected nodes and points
        connected_kf_ids = set()
        connected_kf_point_ids = set()
        # Iterate over all the edges
        for (kf1_id, kf2_id) in self.edges.keys():
            # Check if this node is part of this edge
            # If it is, the other node and its points are of interest
            assert kf1_id != kf2_id
            if kf1_id == kf_id:
                connected_kf_ids.add(kf2_id)
                connected_kf_point_ids.update(self.nodes[kf2_id])
            elif kf2_id == kf_id:
                connected_kf_ids.add(kf1_id)
                connected_kf_point_ids.update(self.nodes[kf1_id])

        return connected_kf_ids, connected_kf_point_ids
    
    def get_neighbor_frames_and_their_points(self, kf_ids: set) -> tuple[set[int], set[int]]:
        """Returns all the neighboring keyframes K2 of a set of keyframes K1, along with their points"""
        neighbors = set()
        neighbor_points = set()
        # Iterate over all the edges
        for (kf1_id, kf2_id) in self.edges.keys():
            # Search for a neighbor
            if kf1_id in kf_ids and kf2_id not in kf_ids:
                neighbors.add(kf2_id)
                neighbor_points.update(self.nodes[kf2_id])
            elif kf1_id not in kf_ids and kf2_id in kf_ids:
                neighbors.add(kf1_id)
                neighbor_points.update(self.nodes[kf1_id])

        return neighbors, neighbor_points
    

    def remove_keyframe(self, kf_id: int):
        """Removes a keyframe from the graph and updates any edges associated with it."""
        if kf_id not in self.nodes:
            log.warning(f"[Graph] Keyframe {kf_id} does not exist.")
            return
        
        self._remove_node(kf_id)
        self._remove_edges_with(kf_id)
        
        self.spanning_tree._remove_node(kf_id)
        self.spanning_tree._remove_edges_with(kf_id)
        
        self.essential_graph._remove_node(kf_id)
        self.essential_graph._remove_edges_with(kf_id)
        
        # Remove loop closure edges that involve the removed keyframe.
        edges_to_remove = [edge for edge in self.loop_closure_edges if kf_id in edge]
        for edge in edges_to_remove:
            del self.loop_closure_edges[edge]

        # Note: In a fully featured system you might want to re-compute or update the spanning tree
        # to ensure full connectivity, but for simplicity we are only removing associated edges here.

    def remove_matches(self, matches: set[int, int]):
        """Remove a set of point <-> node observations"""
        for (pid, kf_id) in matches:
            self.nodes[kf_id].remove(pid)

    def remove_points(self, pids: set[int]):
        """Removes points from graph"""
        for kf_id in self.nodes.keys():
            self.nodes[kf_id] = self.nodes[kf_id] - pids



    def print_graphs(self):
        """
        Prints the current covisibility graph, spanning tree, loop closure edges, and the computed essential graph.
        """
        log.info("=== Covisibility Graph Edges ===")
        for edge, weight in self.edges.items():
            log.info(f"{edge}: {weight}")
        
        log.info("\n=== Spanning Tree Edges ===")
        for edge, weight in self.spanning_tree.edges.items():
            log.info(f"{edge}: {weight}")
        
        log.info("\n=== Loop Closure Edges ===")
        for edge, weight in self.loop_closure_edges.items():
            log.info(f"{edge}: {weight}")
        
        log.info("\n=== Essential Graph Edges ===")
        for edge, weight in self.essential_graph.edges.items():
            log.info(f"{edge}: {weight}")
