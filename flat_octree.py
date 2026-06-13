"""
flat_octree.py
--------------
Builds a flat-array octree from particle positions and velocities,
ready to be shipped to JAX/GPU for Barnes-Hut force queries.

Usage
-----
    from flat_octree import build_flat_tree, FlatTree

    tree, n_nodes = build_flat_tree(pos, vel, mass)
    # tree is a FlatTree namedtuple of numpy arrays, shape (n_nodes, ...)
    # ship to JAX with {k: jnp.array(v) for k, v in tree._asdict().items()}
"""

import numpy as np
from collections import namedtuple

# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

# All per-node data lives in parallel arrays indexed by an integer node_id.
# We pre-allocate MAX_NODES slots and fill them during insertion.
# A node is "allocated" by bumping a global counter; never freed.

# Safe upper bound: a fully-occupied octree over N particles has at most
# ~8*N nodes (each particle insertion adds at most depth+1 nodes, and
# depth ≤ log8(N)).  We use 10*N to be safe against clustered inputs.
def _max_nodes(n_particles: int) -> int:
    return max(64, 10 * n_particles)


# Returned to the caller — slice to [0:n_nodes] before shipping to JAX.
FlatTree = namedtuple("FlatTree", [
    "com_xyz",    # (MAX_NODES, 3)  float64 — center of mass position
    "com_vel",    # (MAX_NODES, 3)  float64 — mass-weighted avg velocity
    "mass",       # (MAX_NODES,)    float64 — total mass in subtree
    "half_w",     # (MAX_NODES,)    float64 — half-width of this node's cube
    "center",     # (MAX_NODES, 3)  float64 — geometric center of cube (not COM)
    "children",   # (MAX_NODES, 8)  int32   — child indices; -1 = no child
    "is_leaf",    # (MAX_NODES,)    bool
    "particle",   # (MAX_NODES,)    int32   — particle index if leaf, else -1
])


# ---------------------------------------------------------------------------
# Builder class (internal)
# ---------------------------------------------------------------------------

class _OctreeBuilder:
    """
    Builds the flat tree via ordinary Python recursion.
    Not called by JAX — runs on CPU before each derivatives() call.
    """

    def __init__(self, n_particles: int):
        M = _max_nodes(n_particles)
        # Allocate all arrays up-front; zero/false/-1 are safe defaults.
        self.com_xyz   = np.zeros((M, 3), dtype=np.float64)
        self.com_vel   = np.zeros((M, 3), dtype=np.float64)
        self.mass      = np.zeros( M,     dtype=np.float64)
        self.half_w    = np.zeros( M,     dtype=np.float64)
        self.center    = np.zeros((M, 3), dtype=np.float64)
        self.children  = np.full ((M, 8), -1, dtype=np.int32)
        self.is_leaf   = np.ones ( M,     dtype=bool)
        self.particle  = np.full ( M,     -1, dtype=np.int32)
        self._next     = 0   # next free slot

    # ------------------------------------------------------------------
    # Slot allocation
    # ------------------------------------------------------------------

    def _alloc(self) -> int:
        """Claim the next free node slot and return its index."""
        idx = self._next
        if idx >= len(self.mass):
            raise RuntimeError(
                f"Octree overflow: needed more than {len(self.mass)} nodes. "
                "Increase the MAX_NODES multiplier in _max_nodes()."
            )
        self._next += 1
        return idx

    # ------------------------------------------------------------------
    # Octant helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _octant(pos_i: np.ndarray, node_center: np.ndarray) -> int:
        """
        Map a 3D position to one of 8 octant indices (0-7).

        Bit layout:
            bit 0 (value 1): x >= center_x
            bit 1 (value 2): y >= center_y
            bit 2 (value 4): z >= center_z

        This gives a unique index for each combination of
        (left/right, front/back, bottom/top).
        """
        return (
            (1 if pos_i[0] >= node_center[0] else 0) |
            (2 if pos_i[1] >= node_center[1] else 0) |
            (4 if pos_i[2] >= node_center[2] else 0)
        )

    @staticmethod
    def _child_center(parent_center: np.ndarray,
                      parent_half_w: float,
                      octant: int) -> np.ndarray:
        """
        Compute the geometric center of a child octant cube.

        Each child is offset by ±quarter_width in each axis.
        The sign is determined by the octant bits:
            bit 0 → x:  +quarter if set, -quarter if not
            bit 1 → y
            bit 2 → z
        """
        q = parent_half_w / 2.0          # quarter-width of parent = half of child
        offset = np.array([
            +q if (octant & 1) else -q,
            +q if (octant & 2) else -q,
            +q if (octant & 4) else -q,
        ])
        return parent_center + offset

    # ------------------------------------------------------------------
    # Core insertion
    # ------------------------------------------------------------------

    def _insert(self,
                node_id: int,
                particle_idx: int,
                pos: np.ndarray,
                vel: np.ndarray,
                mass: np.ndarray) -> None:
        """
        Insert particle `particle_idx` into the subtree rooted at `node_id`.

        Invariant maintained: after every insertion, node_id's .mass,
        .com_xyz, and .com_vel reflect ALL particles in its subtree.
        We update them here incrementally on the way down.
        """

        p_pos  = pos [particle_idx]   # shape (3,)
        p_vel  = vel [particle_idx]   # shape (3,)
        p_mass = mass[particle_idx]   # scalar

        # ── Update this node's aggregate quantities ──────────────────────
        # Running weighted average:
        #   new_com = (old_com * old_mass + p_pos * p_mass) / new_mass
        old_mass = self.mass[node_id]
        new_mass = old_mass + p_mass

        if new_mass > 0.0:
            self.com_xyz[node_id] = (
                self.com_xyz[node_id] * old_mass + p_pos * p_mass
            ) / new_mass
            self.com_vel[node_id] = (
                self.com_vel[node_id] * old_mass + p_vel * p_mass
            ) / new_mass

        self.mass[node_id] = new_mass

        # ── Case 1: empty leaf — just store the particle here ────────────
        if self.particle[node_id] == -1 and self.is_leaf[node_id]:
            self.particle[node_id] = particle_idx
            return

        # ── Case 2: occupied leaf — must subdivide ───────────────────────
        # Create 8 children, then re-insert the existing particle and
        # fall through to insert the new one.
        if self.is_leaf[node_id]:
            self._subdivide(node_id, pos, vel, mass)
            # After subdivision this node is an internal node; continue below.

        # ── Case 3: internal node — recurse into the correct child ───────
        octant   = self._octant(p_pos, self.center[node_id])
        child_id = self.children[node_id, octant]

        if child_id == -1:
            # Child octant doesn't exist yet — allocate it.
            child_id = self._alloc_child(node_id, octant)

        self._insert(child_id, particle_idx, pos, vel, mass)

    def _subdivide(self,
                   node_id: int,
                   pos: np.ndarray,
                   vel: np.ndarray,
                   mass: np.ndarray) -> None:
        """
        Convert a leaf node into an internal node.

        Creates the child slot for the existing particle and re-inserts it.
        Does NOT update mass/com — those are already correct for this node.
        """
        existing_particle = self.particle[node_id]

        # Mark as internal
        self.is_leaf  [node_id] = False
        self.particle [node_id] = -1

        # Find which octant the existing particle belongs to
        existing_pos = pos[existing_particle]
        octant       = self._octant(existing_pos, self.center[node_id])
        child_id     = self._alloc_child(node_id, octant)

        # Re-insert the existing particle into the new child.
        # Note: we call _insert on the child directly to avoid double-counting
        # the mass on this node (it was already counted when first inserted).
        self._insert(child_id, existing_particle, pos, vel, mass)

    def _alloc_child(self, parent_id: int, octant: int) -> int:
        """
        Allocate a new leaf node for the given octant of parent_id.
        Sets the child's geometric center and half-width, registers it
        in parent's children array.
        """
        child_id   = self._alloc()
        child_hw   = self.half_w [parent_id] / 2.0
        child_ctr  = self._child_center(
            self.center[parent_id],
            self.half_w[parent_id],
            octant
        )
        self.half_w[child_id]  = child_hw
        self.center[child_id]  = child_ctr
        self.children[parent_id, octant] = child_id
        return child_id

    # ------------------------------------------------------------------
    # Public build entry point
    # ------------------------------------------------------------------

    def build(self,
              pos:  np.ndarray,
              vel:  np.ndarray,
              mass: np.ndarray) -> int:
        """
        Build the complete flat octree for N particles.

        Parameters
        ----------
        pos  : (N, 3) float64 — particle positions
        vel  : (N, 3) float64 — particle velocities
        mass : (N,)   float64 — particle masses

        Returns
        -------
        n_nodes_used : int — number of valid slots in the flat arrays.
                       Caller should slice all arrays to [:n_nodes_used].
        """
        N = len(pos)
        if N == 0:
            return 0

        # ── Allocate root node ───────────────────────────────────────────
        root_id = self._alloc()   # always 0

        # Root cube: centered on midpoint of positions, half-width large
        # enough to contain all particles with a margin.
        # We deliberately make it slightly larger than tight to avoid
        # boundary issues when particles are near the box edge mid-RK45 stage.
        lo      = pos.min(axis=0)
        hi      = pos.max(axis=0)
        center  = (lo + hi) / 2.0
        half_w  = np.max(hi - lo) / 2.0 * 1.05 + 1e-6  # 5% margin

        self.center[root_id] = center
        self.half_w[root_id] = half_w

        # ── Insert all particles ─────────────────────────────────────────
        for i in range(N):
            self._insert(root_id, i, pos, vel, mass)

        return self._next   # number of slots used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_flat_tree(
        pos:  np.ndarray,
        vel:  np.ndarray,
        mass: np.ndarray,
) -> tuple[FlatTree, int]:
    """
    Build a flat-array octree from particle state.

    Parameters
    ----------
    pos  : (N, 3) float64
    vel  : (N, 3) float64
    mass : (N,)   float64

    Returns
    -------
    tree     : FlatTree namedtuple — arrays of shape (n_nodes, ...).
               Already sliced to used nodes; safe to pass directly to
               {k: jnp.array(v) for k, v in tree._asdict().items()}.
    n_nodes  : int — number of nodes in the tree.
    """
    pos  = np.asarray(pos,  dtype=np.float64)
    vel  = np.asarray(vel,  dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    builder   = _OctreeBuilder(len(pos))
    n_nodes   = builder.build(pos, vel, mass)

    tree = FlatTree(
        com_xyz   = builder.com_xyz  [:n_nodes],
        com_vel   = builder.com_vel  [:n_nodes],
        mass      = builder.mass     [:n_nodes],
        half_w    = builder.half_w   [:n_nodes],
        center    = builder.center   [:n_nodes],
        children  = builder.children [:n_nodes],
        is_leaf   = builder.is_leaf  [:n_nodes],
        particle  = builder.particle [:n_nodes],
    )
    return tree, n_nodes


# ---------------------------------------------------------------------------
# Sanity checks (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N   = 1000

    pos  = rng.standard_normal((N, 3))
    vel  = rng.standard_normal((N, 3))
    mass = rng.uniform(0.5, 2.0, N)

    tree, n_nodes = build_flat_tree(pos, vel, mass)

    print(f"N particles : {N}")
    print(f"Nodes used  : {n_nodes}  (ratio {n_nodes/N:.2f}x)")
    print(f"Tree depth  : ~{np.ceil(np.log(N)/np.log(8)):.0f} levels")

    # ── Check 1: root mass equals sum of all particle masses ────────────
    root_mass = tree.mass[0]
    expected  = mass.sum()
    assert np.isclose(root_mass, expected), \
        f"Root mass mismatch: {root_mass} vs {expected}"
    print(f"Root mass check       : PASSED  ({root_mass:.4f})")

    # ── Check 2: root COM equals particle-weighted average ──────────────
    expected_com = (pos * mass[:, None]).sum(axis=0) / mass.sum()
    assert np.allclose(tree.com_xyz[0], expected_com, atol=1e-10), \
        f"Root COM mismatch: {tree.com_xyz[0]} vs {expected_com}"
    print(f"Root COM check        : PASSED  ({tree.com_xyz[0]})")

    # ── Check 3: every leaf references a valid particle ─────────────────
    leaf_mask     = tree.is_leaf
    leaf_particles = tree.particle[leaf_mask]
    assert (leaf_particles >= 0).all(), "Leaf with no particle index"
    assert len(np.unique(leaf_particles)) == N, \
        f"Expected {N} unique particles at leaves, got {len(np.unique(leaf_particles))}"
    print(f"Leaf particle check   : PASSED  ({leaf_mask.sum()} leaves for {N} particles)")

    # ── Check 4: all children indices are in-bounds ─────────────────────
    valid_children = tree.children[tree.children != -1]
    assert (valid_children < n_nodes).all(), "Child index out of bounds"
    print(f"Children bounds check : PASSED")

    print("\nAll checks passed.")
