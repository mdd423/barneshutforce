"""
tree_to_device.py
-----------------
Transfers a FlatTree (numpy, CPU) onto the JAX device (GPU/TPU/CPU-XLA),
with dtype coercion and validation.

Designed to slot between build_flat_tree() and compute_accels_jax().
"""

import numpy as np
import jax.numpy as jnp
import jax
from flat_octree import build_flat_tree, FlatTree
from typing import NamedTuple


# ---------------------------------------------------------------------------
# The device-side tree type
# ---------------------------------------------------------------------------
# We use a separate namedtuple so type annotations are honest about what
# lives on the device vs the host.  The fields are identical to FlatTree
# but every array is a jnp.ndarray after transfer.

class DeviceTree(NamedTuple):
    com_xyz   : jax.Array   # (n_nodes, 3)  float32 or float64
    com_vel   : jax.Array   # (n_nodes, 3)
    mass      : jax.Array   # (n_nodes,)
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

    # ── Step 1: validate inputs ─────────────────────────────────────────────
    # Catch the most common mistakes before the transfer, not after a silent
    # GPU error.  These checks are cheap and only run on CPU.

    n = len(tree.mass)

    if n == 0:
        raise ValueError("tree_to_device received an empty tree (0 nodes). "
                         "Did build_flat_tree return 0 nodes?")

    # All spatial arrays must agree on node count
    for field_name in ("com_xyz", "com_vel", "half_w", "center",
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

    # ── Step 2: coerce dtypes on CPU before transfer ────────────────────────
    # It is faster to cast on CPU (a memcpy within RAM) than to cast on the
    # GPU after transfer.  We do this explicitly so the dtype decision is
    # visible here, not buried inside jnp.array().

    com_xyz_h  = tree.com_xyz .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    com_vel_h  = tree.com_vel .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    mass_h     = tree.mass    .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    half_w_h   = tree.half_w  .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    center_h   = tree.center  .astype(np.float32 if float_dtype == jnp.float32 else np.float64)
    children_h = tree.children                    # already int32
    is_leaf_h  = tree.is_leaf                     # bool — no cast needed
    particle_h = tree.particle                    # already int32

    # ── Step 3: transfer to device ──────────────────────────────────────────
    # jnp.array() is the host→device boundary.
    # Each call initiates a DMA transfer from RAM to device memory.
    # After this point these arrays are opaque to Python — JAX manages them.
    #
    # We make ONE call per array.  Calling jnp.array() in a tight loop on
    # many tiny arrays is slow due to per-call overhead; 8 calls for 8 fields
    # is fine.

    return DeviceTree(
        com_xyz   = jnp.array(com_xyz_h),    # (n, 3) float
        com_vel   = jnp.array(com_vel_h),    # (n, 3) float
        mass      = jnp.array(mass_h),       # (n,)   float
        half_w    = jnp.array(half_w_h),     # (n,)   float
        center    = jnp.array(center_h),     # (n, 3) float
        children  = jnp.array(children_h),   # (n, 8) int32
        is_leaf   = jnp.array(is_leaf_h),    # (n,)   bool
        particle  = jnp.array(particle_h),   # (n,)   int32
    )


# ---------------------------------------------------------------------------
# Convenience: build + transfer in one call
# ---------------------------------------------------------------------------

def build_and_transfer(
        pos         : np.ndarray,
        vel         : np.ndarray,
        mass        : np.ndarray,
        float_dtype = jnp.float32,
) -> tuple[DeviceTree, int]:
    """
    Convenience wrapper: build the flat octree on CPU and immediately
    transfer it to the JAX device.  This is what derivatives() calls.

    Parameters
    ----------
    pos, vel : (N, 3) float64 numpy arrays from scipy's state vector
    mass     : (N,)   float64 numpy array (constant across the simulation)

    Returns
    -------
    device_tree : DeviceTree  — lives on JAX device, ready for vmap
    n_nodes     : int         — for diagnostics / stack size checks
    """
    flat_tree, n_nodes = build_flat_tree(pos, vel, mass)
    device_tree        = tree_to_device(flat_tree, float_dtype=float_dtype)
    return device_tree, n_nodes


# ---------------------------------------------------------------------------
# How derivatives() uses this
# ---------------------------------------------------------------------------

def derivatives_sketch(t, state, mass_np, theta, softening,
                        compute_accels_jax):
    """
    Sketch of the full derivatives() function showing where
    build_and_transfer fits.  Not runnable without the JAX force kernel.
    """
    N = len(mass_np)
    pos_np = state[:3*N].reshape(N, 3)
    vel_np = state[3*N:].reshape(N, 3)

    # ── CPU: build octree and ship to device ──────────────────────────────
    # This is the only host work per RK45 stage.  Everything below runs
    # on the device with no further Python involvement.
    device_tree, _ = build_and_transfer(pos_np, vel_np, mass_np)

    # ── Device: transfer pos/vel too ──────────────────────────────────────
    # Separate from the tree because pos/vel change every stage while
    # mass_np is constant (could be transferred once at startup).
    pos_jnp  = jnp.array(pos_np)
    vel_jnp  = jnp.array(vel_np)
    mass_jnp = jnp.array(mass_np)

    # ── Device: compute all accelerations in parallel ─────────────────────
    # compute_accels_jax is vmap'd + jit'd — one GPU kernel launch.
    accel_jnp = compute_accels_jax(pos_jnp, vel_jnp, mass_jnp,
                                   device_tree, theta, softening)

    # ── Pull back to CPU for scipy ─────────────────────────────────────────
    # np.array() blocks until the GPU kernel finishes (implicit sync).
    accel_np = np.array(accel_jnp)

    dstate = np.concatenate([vel_np.ravel(), accel_np.ravel()])
    return dstate


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng  = np.random.default_rng(0)
    N    = 500
    pos  = rng.standard_normal((N, 3))
    vel  = rng.standard_normal((N, 3))
    mass = rng.uniform(0.5, 2.0, N)

    print("Building flat tree on CPU...")
    flat_tree, n_nodes = build_flat_tree(pos, vel, mass)
    print(f"  {n_nodes} nodes for {N} particles")
    print(f"  Host memory: "
          f"{sum(a.nbytes for a in flat_tree) / 1024:.1f} KB")

    print("\nTransferring to device (float32)...")
    dtree_f32 = tree_to_device(flat_tree, float_dtype=jnp.float32)
    print(f"  Device memory (float32): "
          f"{sum(a.nbytes for a in dtree_f32) / 1024:.1f} KB")
    print(f"  com_xyz dtype : {dtree_f32.com_xyz.dtype}")
    print(f"  children dtype: {dtree_f32.children.dtype}")
    print(f"  is_leaf dtype : {dtree_f32.is_leaf.dtype}")

    print("\nTransferring to device (float64)...")
    dtree_f64 = tree_to_device(flat_tree, float_dtype=jnp.float64)
    print(f"  Device memory (float64): "
          f"{sum(a.nbytes for a in dtree_f64) / 1024:.1f} KB")

    # Verify root mass survived the transfer
    root_mass_host   = float(flat_tree.mass[0])
    root_mass_device = float(dtree_f32.mass[0])
    assert abs(root_mass_host - root_mass_device) / root_mass_host < 1e-5, \
        "Root mass changed during transfer — dtype precision issue"
    print(f"\nRoot mass preserved across transfer: {root_mass_device:.4f} ✓")
    print(f"DeviceTree is a JAX pytree: {jax.tree_util.tree_leaves(dtree_f32).__class__.__name__} "
          f"with {len(jax.tree_util.tree_leaves(dtree_f32))} leaves")
