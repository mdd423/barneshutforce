import scipy
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple
from typing import NamedTuple
from functools import partial

def _max_nodes(n_particles: int) -> int:
    return max(64, 10 * n_particles)

def unpack(state, N): 
    pos = state[0 : 3*N].reshape(N, 3) 
    vel = state[3*N: 6*N].reshape(N, 3) 
    return pos, vel

def pack(pos, vel): 
    return np.concatenate([pos.flatten(), vel.flatten()])


FlatTree = namedtuple("FlatTree", [
    "com_xyz",    # (MAX_NODES, 3)  float64 — center of mass position
    "coc_vel",    # (MAX_NODES, 3)  float64 — mass-weighted avg velocity
    "mass",       # (MAX_NODES,)    float64 — total mass in subtree
    "charge",       # (MAX_NODES,)    float64 — total mass in subtree
    "half_w",     # (MAX_NODES,)    float64 — half-width of this node's cube
    "center",     # (MAX_NODES, 3)  float64 — geometric center of cube (not COM)
    "children",   # (MAX_NODES, 8)  int32   — child indices; -1 = no child
    "is_leaf",    # (MAX_NODES,)    bool
    "particle",   # (MAX_NODES,)    int32   — particle index if leaf, else -1
])

class _OctreeBuilder:
    """
    Builds the flat tree via ordinary Python recursion.
    Not called by JAX — runs on CPU before each derivatives() call.
    """

    def __init__(self, n_particles: int):
        M = _max_nodes(n_particles)
        # Allocate all arrays up-front; zero/false/-1 are safe defaults.
        self.com_xyz   = np.zeros((M, 3), dtype=np.float64)
        self.coc_vel   = np.zeros((M, 3), dtype=np.float64)
        self.mass      = np.zeros( M,     dtype=np.float64)
        self.charge      = np.zeros( M,     dtype=np.float64)
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
                mass: np.ndarray,
               charge: np.ndarray) -> None:
        """
        Insert particle `particle_idx` into the subtree rooted at `node_id`.

        Invariant maintained: after every insertion, node_id's .mass,
        .com_xyz, and .com_vel reflect ALL particles in its subtree.
        We update them here incrementally on the way down.
        """

        p_pos  = pos [particle_idx]   # shape (3,)
        p_vel  = vel [particle_idx]   # shape (3,)
        p_mass = mass[particle_idx]   # scalar
        p_charge = charge[particle_idx]   # scalar

        # ── Update this node's aggregate quantities ──────────────────────
        # Running weighted average:
        #   new_com = (old_com * old_mass + p_pos * p_mass) / new_mass
        old_mass = self.mass[node_id]
        new_mass = old_mass + p_mass

        old_charge = self.charge[node_id]
        new_charge = old_charge + p_charge

        if new_mass > 0.0:
            self.com_xyz[node_id] = (
                self.com_xyz[node_id] * old_mass + p_pos * p_mass
            ) / new_mass
            self.coc_vel[node_id] = (
                self.coc_vel[node_id] * old_charge + p_vel * p_charge
            ) / new_charge

        self.mass[node_id] = new_mass
        self.charge[node_id] = new_charge

        # ── Case 1: empty leaf — just store the particle here ────────────
        if self.particle[node_id] == -1 and self.is_leaf[node_id]:
            self.particle[node_id] = particle_idx
            return

        # ── Case 2: occupied leaf — must subdivide ───────────────────────
        # Create 8 children, then re-insert the existing particle and
        # fall through to insert the new one.
        if self.is_leaf[node_id]:
            self._subdivide(node_id, pos, vel, mass, charge)
            # After subdivision this node is an internal node; continue below.

        # ── Case 3: internal node — recurse into the correct child ───────
        octant   = self._octant(p_pos, self.center[node_id])
        child_id = self.children[node_id, octant]

        if child_id == -1:
            # Child octant doesn't exist yet — allocate it.
            child_id = self._alloc_child(node_id, octant)

        self._insert(child_id, particle_idx, pos, vel, mass, charge)

    def _subdivide(self,
                   node_id: int,
                   pos: np.ndarray,
                   vel: np.ndarray,
                   mass: np.ndarray,
                  charge: np.ndarray) -> None:
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
        self._insert(child_id, existing_particle, pos, vel, mass, charge)

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
              mass: np.ndarray,
              charge: np.ndarray) -> int:
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
            self._insert(root_id, i, pos, vel, mass, charge)

        return self._next   # number of slots used

def build_flat_tree(
        pos:  np.ndarray,
        vel:  np.ndarray,
        mass: np.ndarray,
        charge: np.ndarray
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
    charge = np.asarray(charge, dtype=np.float64)

    builder   = _OctreeBuilder(len(pos))
    n_nodes   = builder.build(pos, vel, mass, charge)

    tree = FlatTree(
        com_xyz   = builder.com_xyz  [:n_nodes],
        coc_vel   = builder.coc_vel  [:n_nodes],
        mass      = builder.mass     [:n_nodes],
        charge    = builder.charge   [:n_nodes],
        half_w    = builder.half_w   [:n_nodes],
        center    = builder.center   [:n_nodes],
        children  = builder.children [:n_nodes],
        is_leaf   = builder.is_leaf  [:n_nodes],
        particle  = builder.particle [:n_nodes],
    )
    return tree, n_nodes

class DeviceTree(NamedTuple):
    com_xyz   : jax.Array   # (n_nodes, 3)  float32 or float64
    coc_vel   : jax.Array   # (n_nodes, 3)
    mass      : jax.Array   # (n_nodes,)
    charge  : jax.Array
    half_w    : jax.Array   # (n_nodes,)
    center    : jax.Array   # (n_nodes, 3)
    children  : jax.Array   # (n_nodes, 8)  int32  — MUST be int32
    is_leaf   : jax.Array   # (n_nodes,)    bool
    particle  : jax.Array   # (n_nodes,)    int32


# ---------------------------------------------------------------------------
# Core transfer function
# ---------------------------------------------------------------------------

def tree_to_device(
        tree    : FlatTree,
        float_dtype = jnp.float32,   # float32 = GPU speed, float64 = precision
) -> DeviceTree:
    """
    Copy a FlatTree from host (numpy/CPU) onto the JAX device.

    Parameters
    ----------
    tree        : FlatTree returned by build_flat_tree().
                  Already sliced to used nodes — do NOT pass the raw
                  pre-allocated builder arrays.
    float_dtype : jnp.float32 (default, fast on GPU)
                  or jnp.float64 (slower, needed for long integrations
                  where energy drift matters).

    Returns
    -------
    DeviceTree namedtuple of jnp arrays.
    The returned object is a JAX pytree: you can pass it as a single
    argument into jit/vmap'd functions.
    """
    n = len(tree.mass)

    if n == 0:
        raise ValueError("tree_to_device received an empty tree (0 nodes). "
                         "Did build_flat_tree return 0 nodes?")

    # All spatial arrays must agree on node count
    for field_name in ("com_xyz", "coc_vel", "half_w", "center",
                       "children", "is_leaf", "particle"):
        arr = getattr(tree, field_name)
        if len(arr) != n:
            raise ValueError(
                f"Field '{field_name}' has length {len(arr)}, "
                f"expected {n} (from tree.mass)."
            )

    # children must be int32; int64 causes silent indexing bugs on GPU
    if tree.children.dtype != np.int32:
        raise TypeError(
            f"tree.children must be int32, got {tree.children.dtype}. "
            "Check your build_flat_tree output."
        )

    # Sanity: root node (index 0) must have positive mass
    if tree.mass[0] <= 0.0:
        raise ValueError(
            f"Root node mass is {tree.mass[0]:.3e}. "
            "Tree was built with no particles, or mass array is wrong."
        )

   

    com_xyz_h  = tree.com_xyz .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    coc_vel_h  = tree.coc_vel .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    mass_h     = tree.mass    .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    charge_h     = tree.charge    .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    half_w_h   = tree.half_w  .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    center_h   = tree.center  .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    children_h = tree.children                    # already int32
    is_leaf_h  = tree.is_leaf                     # bool — no cast needed
    particle_h = tree.particle                    # already int32

   
    return DeviceTree(
        com_xyz   = jnp.array(com_xyz_h),    # (n, 3) float
        coc_vel   = jnp.array(coc_vel_h),    # (n, 3) float
        mass      = jnp.array(mass_h),       # (n,)   float
        charge      = jnp.array(charge_h),       # (n,)   float
        half_w    = jnp.array(half_w_h),     # (n,)   float
        center    = jnp.array(center_h),     # (n, 3) float
        children  = jnp.array(children_h),   # (n, 8) int32
        is_leaf   = jnp.array(is_leaf_h),    # (n,)   bool
        particle  = jnp.array(particle_h),   # (n,)   int32
    )

class BarnesHutSimulator:

    def __init__(self, mass, charge, force_fn,
                 theta=0.5, softening=1e-3,
                 max_stack=128, float_dtype=jnp.float32):
        """
        Parameters
        ----------
        mass      : (N,) numpy array — constant across simulation
        charge    : (N,) numpy array — or any per-particle scalar property
        force_fn  : callable with signature
                        force_fn(r, r_vec, mass_node, charge_node,
                                 mass_i, charge_i, com_vel_node, vel_i)
                        → (3,) jnp array (acceleration on particle i)
                    Must use only JAX ops — no Python if on traced values.
        """
        self.mass        = np.asarray(mass,   dtype=np.float64)
        self.charge      = np.asarray(charge, dtype=np.float64)
        self.N           = len(mass)
        self.theta       = theta
        self.softening   = softening
        self.float_dtype = float_dtype

        # Ship constant arrays to device once at construction time
        self.mass_jnp   = jnp.array(self.mass,   dtype=float_dtype)
        self.charge_jnp = jnp.array(self.charge, dtype=float_dtype)

        # Bake force_fn into the traversal at jit-compile time.
        # Each simulator instance gets its own compiled XLA program.
        _force_on_one = partial(
            _force_on_one_template,
            force_fn  = force_fn,
            max_stack = max_stack,
        )
        _force_on_one_jit = jax.jit(_force_on_one)

        self._force_all = jax.vmap(
            _force_on_one_jit,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None)
            #         i  pos vel mass chg  tree  θ    ε
        )

    def derivatives(self, t, state):
        """
        Drop-in for scipy solve_ivp. Builds the octree on CPU,
        ships to device, runs vmap'd force kernel, returns to CPU.
        """
        pos_np = state[:3*self.N].reshape(self.N, 3)
        vel_np = state[3*self.N:].reshape(self.N, 3)

        # CPU: build flat octree from current positions + velocities
        flat_tree, _ = build_flat_tree(pos_np, vel_np, self.mass,self.charge)
        device_tree  = tree_to_device(flat_tree, float_dtype=self.float_dtype)

        # Transfer pos/vel to device (mass/charge already there)
        pos_jnp = jnp.array(pos_np, dtype=self.float_dtype)
        vel_jnp = jnp.array(vel_np, dtype=self.float_dtype)
        ids_jnp = jnp.arange(self.N, dtype=jnp.int32)

        accel_jnp = self._force_all(
            ids_jnp, pos_jnp, vel_jnp,
            self.mass_jnp, self.charge_jnp, jnp.array(t),
            device_tree, self.theta, self.softening
        )

        accel_np = np.array(accel_jnp)
        return np.concatenate([vel_np.ravel(), accel_np.ravel()])


# ---------------------------------------------------------------------------
# Traversal template — force_fn is baked in via partial, never a traced arg
# ---------------------------------------------------------------------------

def _force_on_one_template(i, pos_i, vel_i, mass_i, charge_i, time,
                            tree, theta, softening,
                            force_fn, max_stack):   # ← partial fills these
    init = (
        jnp.zeros(3),
        jnp.zeros(max_stack, dtype=jnp.int32).at[0].set(0),
        jnp.int32(1),
    )

    def cond(state):
        _, _, sp = state
        return sp > 0

    def body(state): 
        accel, stack, sp = state 
        sp -= 1 
        node = stack[sp] # pop current node
        r_vec = pos_i - tree.com_xyz[node]
        r = jnp.sqrt(jnp.dot(r_vec, r_vec) + softening**2) 
        ratio = 2.0 * tree.half_w[node] / r
        
        is_self = tree.is_leaf[node] & (tree.particle[node] == i) # skip self-interaction 
        approx = tree.is_leaf[node] | (ratio < theta)
        # --- Compute contribution if approximating (or exact leaf) --- 
        force_total = force_fn(r, pos_i, tree.mass[node], mass_i, tree.charge[node], charge_i, r_vec, tree.coc_vel[node], vel_i, time)
        contrib = jnp.where(approx & ~is_self, force_total, jnp.zeros(3)) 
        accel += contrib
        # --- Push children onto stack if not approximating --- 
        should_recurse = ~approx & ~is_self 
        for c in range(8): # unrolled — static count 
            child = tree.children[node, c] 
            valid = should_recurse & (child != -1) 
            stack = jnp.where(valid, stack.at[sp].set(child), stack) 
            sp += jnp.where(valid, 1, 0)
        return accel, stack, sp
    
    accel_final, _, _ = jax.lax.while_loop(cond, body, init)
    return accel_final