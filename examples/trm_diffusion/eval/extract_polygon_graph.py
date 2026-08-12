"""
Vendored VERBATIM from the paper's own repo (kariander1/visual-geo-solver,
scripts/extract_polygon_graph.py — also present locally at
examples/visual-geo-solver/scripts/extract_polygon_graph.py). Not a
reimplementation: this is a literal copy of PolygonGraphExtractor, kept
byte-for-byte identical to the original algorithm (only this header comment,
the debug-visualization method, and the `main`/argparse CLI entrypoint were
dropped — pure scaffolding, no algorithmic content) so eval/polygon_eval.py
calls the *exact* algorithm the paper's own evaluation uses.

One signature change from the original: extract_polygon_from_points now also
returns `vertex_indices` (the point-index cycle order) as a 5th tuple
element. The original computes this internally
(_order_vertices_for_polygon's second return value) but only forwards the
ordered pixel *coordinates*, not the indices into the input `gt_points` list
— we need the indices to compare the recovered polygon's vertex order
against the dataset's known-optimal order (PolygonDataset.optimal_order_for);
recovering them by matching pixel coordinates back to gt_points would be
lossy whenever two points round to the same pixel. No other line changed.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from shapely.geometry import Polygon as ShapelyPolygon
import networkx as nx


class PolygonGraphExtractor:
    def __init__(
        self,
        coverage_threshold: float = 0.9,
        edge_width: int = 2,
        debug: bool = False
    ):
        """
        Initialize the polygon extractor.

        Args:
            coverage_threshold: Minimum fraction of line pixels that must be covered for edge detection (0.9 = 90%)
            edge_width: Expected width of edges in pixels (used for line sampling)
            debug: Whether to show debug visualizations
        """
        self.coverage_threshold = coverage_threshold
        self.edge_width = edge_width
        self.debug = debug

    def extract_polygon_from_points(
        self,
        binary_img: np.ndarray,
        gt_points: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[int, int]], float, float, List[Tuple[int, int]], List[int]]:
        """
        Extract polygon structure from binary image using ground truth points.

        Args:
            binary_img: Image with format: background=0, edges=127, interior=-127 (or normalized: bg=0, edges=1, interior=-1)
            gt_points: Ground truth points in normalized coordinates [0,1]

        Returns:
            Tuple of (vertices, area, perimeter, edges, vertex_indices) where:
            - vertices: List of (x, y) pixel coordinates of vertices in polygon
            - area: Polygon area in normalized coordinates [0,1]
            - perimeter: Polygon perimeter in normalized coordinates [0,1]
            - edges: List of (i, j) tuples indicating edges between vertices
            - vertex_indices: indices into gt_points giving the cycle order
        """
        # New format: background=0, edges=127, interior=-127
        # Use threshold for robustness instead of exact match
        edge_mask = (binary_img > 64)  # Values closer to 127 than to 0 or -127

        # For edge detection, we only care about the edge pixels
        binary_mask = edge_mask

        # Convert GT points to pixel coordinates
        image_size = binary_img.shape[0]
        pixel_vertices = []
        for i, (x, y) in enumerate(gt_points):
            px = int(x * (image_size - 1))
            py = int(y * (image_size - 1))
            pixel_vertices.append((px, py))

        # Detect edges using line coverage
        edges = self._detect_edges_by_coverage(binary_mask, pixel_vertices)

        # Check for self-intersecting edges using original GT coordinates
        has_intersections = self._check_for_intersecting_edges(gt_points, edges)

        # Order vertices to form a polygon
        if has_intersections:
            # Return empty result - invalid polygonization
            return [], 0.0, 0.0, edges, []

        ordered_vertices, vertex_indices = self._order_vertices_for_polygon(pixel_vertices, edges)

        # Calculate area and perimeter using original GT coordinates
        if len(ordered_vertices) >= 3 and vertex_indices:
            # Use original GT coordinates for accurate area calculation
            ordered_gt_coords = [gt_points[i] for i in vertex_indices]
            area, perimeter = self._calculate_polygon_metrics_from_coords(ordered_gt_coords)
        else:
            area, perimeter = 0.0, 0.0

        return ordered_vertices, area, perimeter, edges, vertex_indices

    def _detect_edges_by_coverage(
        self,
        binary_mask: np.ndarray,
        vertices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Detect edges by checking line coverage between all pairs of vertices.

        Args:
            binary_mask: Boolean mask where True = polygon pixels
            vertices: List of (x, y) pixel coordinates

        Returns:
            List of (i, j) tuples indicating edges between vertex i and vertex j
        """
        edges = []
        n_vertices = len(vertices)

        # Check all possible edges in the complete graph
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if self._is_edge_covered(binary_mask, vertices[i], vertices[j]):
                    edges.append((i, j))

        return edges

    def _check_for_intersecting_edges(
        self,
        vertices: List[Tuple[float, float]],
        edges: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if any edges intersect with each other (except at shared vertices).

        Args:
            vertices: List of (x, y) normalized coordinates [0,1]
            edges: List of (i, j) tuples indicating edges between vertex i and vertex j

        Returns:
            True if any edges intersect in their interiors
        """
        if len(edges) <= 1:
            return False

        # Check all pairs of edges for intersections
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if self._edges_intersect(vertices, edges[i], edges[j]):
                    return True

        return False

    def _edges_intersect(
        self,
        vertices: List[Tuple[float, float]],
        edge1: Tuple[int, int],
        edge2: Tuple[int, int]
    ) -> bool:
        """
        Check if two edges intersect (except at shared vertices).

        Args:
            vertices: List of vertex coordinates (normalized [0,1])
            edge1: (i, j) tuple for first edge
            edge2: (k, l) tuple for second edge

        Returns:
            True if edges intersect in their interiors
        """
        i, j = edge1
        k, l = edge2

        # Skip if edges share a vertex
        if i == k or i == l or j == k or j == l:
            return False

        # Get coordinates
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        x3, y3 = vertices[k]
        x4, y4 = vertices[l]

        # Use line intersection formula
        # Line 1: (x1,y1) to (x2,y2)
        # Line 2: (x3,y3) to (x4,y4)

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Lines are parallel (lenient threshold for angles up to ~3 degrees)
        if abs(denom) < 0.05:
            return False

        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection occurs within both line segments
        return 0 < t < 1 and 0 < u < 1

    def _is_edge_covered(
        self,
        binary_mask: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> bool:
        """
        Check if an edge between two points is covered in the binary mask.

        More lenient approach similar to Steiner extractor:
        - Skip endpoints to avoid vertex disc overlap
        - Use small patch around each pixel to account for edge width
        """
        x1, y1 = point1
        x2, y2 = point2

        # Get all pixel coordinates along the line using Bresenham's algorithm
        line_points = self._get_line_points(x1, y1, x2, y2)
        n = len(line_points)

        if n == 0:
            return False

        # Skip endpoints where vertex discs might overlap edges
        # Use a more conservative skip distance for polygons
        skip = max(1, self.edge_width)
        start = min(skip, n)
        end = max(start, n - skip)
        segment = line_points[start:end]

        if len(segment) <= 2:
            # If the edge is extremely short (within vertices), accept it
            return True

        # Sample a small patch around each segment pixel to account for edge width and gaps
        r = 1  # Patch radius - check immediate neighbors
        H, W = binary_mask.shape

        covered_count = 0
        for x, y in segment:
            if 0 <= x < W and 0 <= y < H:
                # Check a small patch around the pixel for more tolerance
                x1p = max(0, x - r)
                y1p = max(0, y - r)
                x2p = min(W, x + r + 1)
                y2p = min(H, y + r + 1)

                # If any pixel in the patch is covered, count it
                if np.any(binary_mask[y1p:y2p, x1p:x2p]):
                    covered_count += 1

        coverage_ratio = covered_count / len(segment)
        is_edge = coverage_ratio >= self.coverage_threshold

        return is_edge

    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """
        Get all pixel coordinates along a line using Bresenham's line algorithm.
        """
        points = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            points.append((x, y))

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def _order_vertices_for_polygon(
        self,
        vertices: List[Tuple[int, int]],
        edges: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Order vertices to form a polygon by finding a cycle in the graph.

        Returns:
            Tuple of (ordered_vertices, vertex_indices) where vertex_indices
            contains the indices of vertices in the cycle order
        """
        if len(vertices) < 3 or len(edges) == 0:
            return [], []

        # Create graph from edges
        G = nx.Graph()
        G.add_nodes_from(range(len(vertices)))
        G.add_edges_from(edges)

        # Find any cycle in the graph using simple DFS
        cycle = self._find_simple_cycle(G)
        if cycle:
            # Convert vertex indices to actual coordinates and return both
            ordered_vertices = [vertices[i] for i in cycle]
            return ordered_vertices, cycle

        return [], []

    def _find_simple_cycle(self, G: nx.Graph) -> Optional[List[int]]:
        """
        Find a Hamiltonian cycle (visiting all vertices) using simple DFS.
        """
        n = G.number_of_nodes()
        if n < 3:
            return None

        def dfs(path: List[int], visited: set) -> Optional[List[int]]:
            if len(path) == n:
                # Check if we can return to start to complete the cycle
                if path[0] in G.neighbors(path[-1]):
                    return path
                return None

            current = path[-1]
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)
                    result = dfs(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(neighbor)

            return None

        # Try starting from each vertex
        for start_vertex in G.nodes():
            result = dfs([start_vertex], {start_vertex})
            if result is not None:
                return result

        return None

    def _calculate_polygon_metrics_from_coords(self, coords: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate polygon area and perimeter directly from normalized coordinates.

        Args:
            coords: List of (x, y) normalized coordinates [0,1]

        Returns:
            Tuple of (area, perimeter) in normalized coordinates [0,1]
        """
        if len(coords) < 3:
            return 0.0, 0.0

        try:
            # Create shapely polygon directly from normalized coordinates
            polygon = ShapelyPolygon(coords)

            # Handle invalid polygons
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Fix self-intersections

            area = float(polygon.area)
            perimeter = float(polygon.length)

        except Exception:
            # Fallback to simple calculation
            area = self._calculate_area_shoelace(coords)
            perimeter = self._calculate_perimeter_euclidean(coords)

        return area, perimeter

    def _calculate_area_shoelace(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0

        n = len(vertices)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def _calculate_perimeter_euclidean(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter using Euclidean distances."""
        if len(vertices) < 2:
            return 0.0

        perimeter = 0.0
        n = len(vertices)

        for i in range(n):
            j = (i + 1) % n
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            perimeter += np.sqrt(dx*dx + dy*dy)

        return perimeter
