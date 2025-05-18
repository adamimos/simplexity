# Data Structures

The library includes a few simple data structures implemented with Equinox to ensure compatibility with JAX transformations such as `jit` and `vmap`. These structures are helpful when writing algorithms that maintain state inside compiled functions.

## Queue

Implemented in `simplexity/data_structures/queue.py`. Supports `enqueue`, `dequeue`, `peek` and `clear` operations. Internally uses two `Stack` instances so that pushing and popping remain efficient under JAX.

## Stack

Located at `simplexity/data_structures/stack.py`. Provides `push`, `pop`, `peek` and `clear`. A `max_size` parameter limits the number of stored elements which makes it suitable for static allocation within JAX functions.

## Heap

Implemented in `simplexity/data_structures/heap.py`. Offers push and pop operations for maintaining a priority queue. Useful for beam search style algorithms.

All structures store JAX arrays or PyTree elements and can therefore be used inside `eqx.filter_jit` or `jax.jit` compiled code.
