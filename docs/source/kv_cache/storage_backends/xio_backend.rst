XIO Backend - Multi-Level KV Cache
====================================

Overview
--------

The XIO Backend provides a **truly hierarchical multi-level KV cache** for LMCache.
It wraps multiple storage backends (CPU, Disk, GDS, Remote, etc.) into a single
unified backend with automatic **cross-level data promotion to CPU**.

In the default LMCache setup, storage backends are independent layers -- if CPU misses
but Disk hits, the data is served from Disk but **not promoted back to CPU**. The XIO
Backend changes this: data found at a slower level is automatically promoted to the
CPU level (level 0), turning the backends into a true cache hierarchy.

Architecture
------------

::

    +--------------------------------------------------+
    |                   CacheEngine                     |
    +--------------------------------------------------+
                          |
                          v
    +--------------------------------------------------+
    |                StorageManager                     |
    |  (minimal changes: duck-typed pipelined get)     |
    +--------------------------------------------------+
                          |
                          v
    +--------------------------------------------------+
    |                  XIOBackend                       |
    |  (AllocatorBackendInterface)                     |
    |                                                  |
    |  +--------------------------------------------+  |
    |  | Level 0: LocalCPUBackend (fastest, small)  |  |
    |  |          <- promotion target               |  |
    |  +--------------------------------------------+  |
    |                    |                             |
    |  +--------------------------------------------+  |
    |  | Level 1: LocalDiskBackend (medium, medium) |  |
    |  +--------------------------------------------+  |
    |                    |                             |
    |  +--------------------------------------------+  |
    |  | Level 2: GdsBackend / Remote (slow, large) |  |
    |  +--------------------------------------------+  |
    +--------------------------------------------------+

Data Flow
---------

**Batched Lookup (batched_contains)**::

    Request: batched_contains([k0, k1, k2, k3, k4])
        |
        +-> Level 0: prefix hit 2 (k0, k1)
        +-> remaining [k2, k3, k4]
        |   Level 1: prefix hit 2 (k2, k3)
        +-> remaining [k4] -> miss everywhere
        +-> total prefix hits = 4

**Batched Get with Concurrent Cross-Level Dispatch**::

    _build_block_mapping([k0..k3]):
        Level 0: [k0, k1]  (indices 0, 1)
        Level 1: [k2, k3]  (indices 2, 3)

    ThreadPool dispatch (concurrent):
        Thread A: Level 0.batched_get_blocking([k0, k1])
        Thread B: Level 1.batched_get_blocking([k2, k3])
                      |
                      +-> Promote [k2, k3] to Level 0 (CPU only)

    Merge results by original index -> [obj0, obj1, obj2, obj3]

**Per-Key Pipeline (fan-out for default backends)**::

    When sub-backend uses default per-key get (e.g. LocalCPUBackend):
    each key is submitted independently to the thread pool.

    Thread 1: get(k0) -> callback(k0) -> to_gpu(k0)
    Thread 2:    get(k1) -> callback(k1) -> to_gpu(k1)     (overlaps)
    Thread 3:       get(k2) -> callback(k2) -> to_gpu(k2)  (overlaps)

    When sub-backend has real batched impl (e.g. GDS):
    whole batch is forwarded as-is; callback fires sequentially.

**Promotion Strategy**::

    Level 0 (CPU): ---- miss ---->  Level 1 (Disk): hit!
                                       |
                   <-- promote --------+
                   (write to CPU only, not to other levels)

    Level 0 (CPU): ---- miss ---->  Level 2 (GDS): hit!
                                       |
                   <-- promote --------+
                   (write to CPU only, skip Level 1)

Configuration
-------------

Enable XIO Backend via YAML configuration:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0
    local_disk: /tmp/lmcache_disk
    max_local_disk_size: 50.0
    enable_xio_backend: true

Or via environment variable:

.. code-block:: bash

    export LMCACHE_ENABLE_XIO_BACKEND=true

Using Mooncake with XIO
~~~~~~~~~~~~~~~~~~~~~~~

Mooncake is integrated into LMCache as a ``RemoteBackend`` via the
``mooncakestore://`` URL scheme.  Since ``RemoteBackend`` overrides
``batched_get_blocking``, XIO automatically dispatches Mooncake through
the **batched** path -- no additional code changes are needed.

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0
    remote_url: "mooncakestore://127.0.0.1:50051/"
    remote_serde: "naive"
    enable_xio_backend: true
    extra_config:
      save_chunk_meta: false
      local_hostname: "localhost"
      metadata_server: "http://127.0.0.1:8005/metadata"
      protocol: "tcp"
      master_server_address: "localhost:50051"

With this configuration, XIO wraps both ``LocalCPUBackend`` and
``RemoteBackend`` (Mooncake) into a two-level hierarchy.  Cache misses
at the CPU level fall through to Mooncake, and retrieved data is
automatically promoted back to CPU.

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    enable_xio_backend: true
    extra_config:
      xio_write_policy: "write_through"   # or "write_back"
      xio_get_workers: 8                  # thread pool size

- **write_through** (default): Data written to all levels on ``put``.
- **write_back**: Only Level 0 receives writes; lower levels populated
  through eviction or explicit flush.
- **xio_get_workers**: Thread pool size for concurrent cross-level gets
  and per-key fan-out.  Defaults to ``max(4, num_levels * 2)``.

Custom Backend Ordering
~~~~~~~~~~~~~~~~~~~~~~~

By default, XIO arranges backends in the order they are created
(CPU → Disk → GDS → Remote → plugins).  You can override this with
``extra_config.xio_backend_order``:

.. code-block:: yaml

    enable_xio_backend: true
    extra_config:
      xio_backend_order:
        - LocalCPUBackend
        - RemoteBackend
        - LocalDiskBackend

The list specifies backends from fastest (level 0) to slowest.
Backend names must match the class names used by LMCache
(``LocalCPUBackend``, ``LocalDiskBackend``, ``GdsBackend``,
``RemoteBackend``, etc.).  Backends not listed are appended at the
end in their default creation order.

Feature Toggle
~~~~~~~~~~~~~~

When ``enable_xio_backend`` is ``false`` (the default), LMCache behaves
exactly as before.  No code path in ``cache_engine`` or
``storage_manager`` is altered.

Key Design Decisions
--------------------

**CPU-Only Promotion**
    When data is found at level N > 0, it is promoted **only to the CPU
    level** (level 0).  Intermediate levels (disk, GDS) are not written
    to during promotion.  This avoids unnecessary write amplification
    while ensuring the hottest data lands in the fastest tier.

**Batch Preservation**
    All batch operations are forwarded to sub-backends as whole batches.
    Sub-backends can apply their own batch optimisations (e.g. GDS
    thread-pool reads, RDMA batching).

**Adaptive Per-Key Fan-Out**
    ``_has_real_batched_get()`` detects whether a sub-backend overrides
    ``batched_get_blocking``.  If not (default per-key loop), XIO fans
    out individual keys to the thread pool so that get + to_gpu of
    different keys overlap in a pipeline.

**Minimal Changes to Original Files**
    - ``cache_engine.py``: one ``elif`` branch + one self-contained method.
    - ``storage_manager.py``: one new method (``batched_get_pipelined``)
      using duck typing (``hasattr``) with no XIO-specific imports.
    - ``config.py``: one config field.
    - ``__init__.py``: one import + wrapping logic at the end.

**Ref Count Correctness**
    - Sub-backend ``get_blocking`` does ``ref_count_up`` → caller holds
      one ref.
    - Promotion calls ``batched_submit_put_task`` → sub-backend does
      ``ref_count_up`` internally.
    - ``cache_engine.retrieve`` does ``ref_count_down`` after ``to_gpu``.
    - Net effect: CPU hot_cache holds its own ref; caller's ref is
      released after use.  No leaks, no double-frees.

Implementation Files
--------------------

- ``lmcache/v1/storage_backend/xio_backend.py`` -- XIO Backend
- ``lmcache/v1/storage_backend/__init__.py`` -- factory wrapping
- ``lmcache/v1/storage_backend/storage_manager.py`` --
  ``batched_get_pipelined`` (duck-typed)
- ``lmcache/v1/cache_engine.py`` -- pipelined retrieve path
- ``lmcache/v1/config.py`` -- ``enable_xio_backend``
- ``tests/v1/storage_backend/test_xio_backend.py`` -- test suite
