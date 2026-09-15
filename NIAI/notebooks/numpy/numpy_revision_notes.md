# NumPy — Complete Revision Notes

A full sweep of NumPy: from array basics to the internals you usually forget. Use this top-to-bottom, or jump to a section for a quick refresher.

---

## 1. What NumPy Is & Why It Matters

- NumPy = **Num**erical **Py**thon. Core library for numerical computing.
- Provides the `ndarray` (n-dimensional array) — a fixed-type, contiguous-memory data structure.
- Faster than Python lists because:
  - Elements are stored in contiguous memory blocks (not scattered like list objects).
  - Operations are **vectorized** — run in compiled C loops, not Python-level loops.
  - No per-element type checking (fixed dtype).
- Foundation for pandas, scikit-learn, TensorFlow, PyTorch, Matplotlib, SciPy.

```python
import numpy as np
```

---

## 2. Creating Arrays

### 2.1 From existing data
```python
np.array([1, 2, 3])                # 1D
np.array([[1, 2], [3, 4]])         # 2D
np.array([1, 2, 3], dtype=float)   # force dtype
```

### 2.2 Built-in generators
```python
np.zeros((3, 4))            # array of 0s
np.ones((2, 3))             # array of 1s
np.full((2, 2), 7)          # filled with a constant
np.empty((2, 2))            # uninitialized (garbage values, fast)
np.eye(4)                   # identity matrix
np.identity(4)              # same as eye for square
```

### 2.3 Ranges & sequences
```python
np.arange(0, 10, 2)         # like range(): [0,2,4,6,8]
np.linspace(0, 1, 5)        # 5 evenly spaced points between 0 and 1 (inclusive)
```

### 2.4 Random arrays
```python
np.random.rand(3, 2)             # uniform [0,1), shape (3,2)
np.random.randn(3, 2)            # standard normal
np.random.randint(0, 10, (3,3))  # random ints in [0,10)
np.random.seed(42)               # reproducibility
np.random.choice([1,2,3], size=5, replace=True)

# Modern API (preferred going forward):
rng = np.random.default_rng(seed=42)
rng.random((3,3))
rng.integers(0, 10, size=5)
```

### 2.5 Copying vs generating like another array
```python
np.zeros_like(arr)
np.ones_like(arr)
np.full_like(arr, 5)
arr.copy()          # deep copy (important — see Section 10)
```

---

## 3. Array Attributes (inspecting an array)

```python
arr.ndim        # number of dimensions
arr.shape       # tuple of dimensions
arr.size        # total number of elements
arr.dtype       # data type of elements
arr.itemsize    # bytes per element
arr.nbytes      # total bytes = size * itemsize
arr.T           # transpose
arr.data        # buffer object (rarely used directly)
```

---

## 4. Data Types (dtype)

- Common dtypes: `int8/16/32/64`, `uint8...`, `float16/32/64`, `bool`, `complex64/128`, `object`, `<U10` (unicode string).
- NumPy arrays are **homogeneous** — all elements share one dtype.
- Type conversion:
```python
arr.astype(np.float32)
arr.astype('int32')
```
- Type promotion: mixing int + float → float; int + str → object/error depending on context.
- Check type: `arr.dtype`, `np.issubdtype(arr.dtype, np.integer)`

---

## 5. Indexing & Slicing

### 5.1 Basic indexing
```python
a[0]          # first element
a[-1]         # last element
a[1:4]        # slice
a[::2]        # every 2nd element
a[::-1]       # reversed
```

### 5.2 2D indexing
```python
a[1, 2]       # row 1, col 2
a[1]          # entire row 1
a[:, 1]       # entire column 1
a[0:2, 1:3]   # sub-matrix
```

### 5.3 Fancy indexing
```python
a[[0, 2, 4]]           # select specific rows/elements by index list
a[[0,1], [1,2]]        # picks (0,1) and (1,2) — paired coordinates
```

### 5.4 Boolean masking
```python
a[a > 5]                 # elements greater than 5
a[(a > 2) & (a < 8)]     # combine conditions with & | ~ (NOT and/or)
a[a > 5] = 0              # conditional assignment
np.where(a > 5, 1, 0)     # vectorized if-else
```

### 5.5 Views vs copies (CRITICAL)
- **Basic slicing → returns a view** (shares memory with original).
- **Fancy indexing / boolean masking → returns a copy**.
```python
b = a[1:3]      # view — modifying b changes a
c = a[[1,3]]    # copy — modifying c does NOT change a
a.base          # None if owns data, else points to original array
```

---

## 6. Reshaping & Manipulating Shape

```python
a.reshape(2, 3)             # reshape (must preserve total size)
a.reshape(-1, 1)            # -1 = "infer this dimension"
a.flatten()                 # returns a copy, 1D
a.ravel()                   # returns a view (if possible), 1D — faster
a.T / a.transpose()         # transpose axes
np.transpose(a, axes=(1,0,2))  # custom axis order for nD

a.resize((3,3))             # in-place resize (pads with zeros/truncates)
np.expand_dims(a, axis=0)   # add a new axis
a[:, np.newaxis]            # same, via indexing
np.squeeze(a)               # remove axes of length 1
```

### Joining arrays
```python
np.concatenate([a, b], axis=0)
np.vstack([a, b])       # stack vertically (row-wise)
np.hstack([a, b])       # stack horizontally (column-wise)
np.column_stack([a, b]) # stack 1D arrays as columns
np.stack([a, b], axis=0)  # creates NEW dimension (unlike concatenate)
```

### Splitting arrays
```python
np.split(a, 3)            # split into 3 equal parts
np.hsplit(a, 2)
np.vsplit(a, 2)
np.array_split(a, 3)      # allows uneven splits
```

---

## 7. Array Operations (Vectorized Math)

### 7.1 Element-wise arithmetic
```python
a + b, a - b, a * b, a / b, a // b, a % b, a ** 2
np.add(a, b), np.subtract(a, b), np.multiply(a, b), np.divide(a, b)
```
Note: `*` is element-wise, **not** matrix multiplication.

### 7.2 Matrix multiplication
```python
a @ b                 # matrix mult operator
np.matmul(a, b)
np.dot(a, b)           # dot product (matmul for 2D, sum-product for 1D)
np.inner(a, b)
np.outer(a, b)
```

### 7.3 Universal functions (ufuncs) — element-wise, vectorized
```python
np.sqrt(a), np.exp(a), np.log(a), np.log2(a), np.log10(a)
np.sin(a), np.cos(a), np.tan(a)
np.abs(a), np.round(a, 2), np.floor(a), np.ceil(a)
np.power(a, 3)
np.clip(a, min_val, max_val)   # bound values into a range
```

### 7.4 Comparison operators (return boolean arrays)
```python
a > b, a == b, a != b
np.array_equal(a, b)        # True/False for whole arrays
np.allclose(a, b)           # tolerant float comparison
```

### 7.5 Aggregate / reduction functions
```python
a.sum(), a.min(), a.max(), a.mean(), a.std(), a.var()
a.sum(axis=0)     # column-wise sum (2D)
a.sum(axis=1)     # row-wise sum
a.argmin(), a.argmax()      # index of min/max
a.cumsum(), a.cumprod()     # running totals
np.median(a)
np.percentile(a, 90)
np.ptp(a)                   # max - min ("peak to peak")
```
Remember: **axis=0 collapses rows (operates down columns), axis=1 collapses columns (operates across rows)**.

---

## 8. Broadcasting

Rule: when operating on two arrays, NumPy compares shapes **from the trailing dimension**:
1. Dimensions are compatible if equal, OR one of them is 1.
2. Missing dimensions are treated as 1.

```python
a = np.array([[1,2,3],[4,5,6]])   # shape (2,3)
b = np.array([10,20,30])          # shape (3,)
a + b        # b is broadcast across each row → works

c = np.array([[1],[2]])           # shape (2,1)
a + c        # c broadcast across columns → works
```
- Broadcasting avoids explicit loops/copies — key to NumPy's speed.
- Shape mismatch that can't broadcast → `ValueError: operands could not be broadcast together`.

---

## 9. Linear Algebra (np.linalg)

```python
np.linalg.det(a)          # determinant
np.linalg.inv(a)          # inverse
np.linalg.pinv(a)         # pseudo-inverse
np.linalg.solve(A, b)     # solve Ax = b
np.linalg.eig(a)          # eigenvalues & eigenvectors
np.linalg.svd(a)          # singular value decomposition
np.linalg.norm(a)         # vector/matrix norm
np.linalg.matrix_rank(a)
np.trace(a)               # sum of diagonal elements
```

---

## 10. Memory: Views, Copies & `id()`

- `b = a` → **same object**, no copy at all.
- `b = a.view()` / basic slicing → **view**, shares data buffer.
- `b = a.copy()` → **independent copy**.
- Check sharing: `np.shares_memory(a, b)`
- This matters a lot in practice — silent bugs happen when you think you copied but only sliced.

```python
a = np.array([1,2,3])
b = a[:2]          # view
b[0] = 99
print(a)           # [99, 2, 3]  <- a changed too!
```

---

## 11. Handling Missing / Special Values

```python
np.nan                     # "Not a Number" — float only
np.isnan(a)                 # boolean mask of NaNs
np.nan_to_num(a)            # replace NaN with 0 (and inf with large numbers)
np.isinf(a)
np.isfinite(a)

np.nansum(a), np.nanmean(a), np.nanmax(a)   # ignore NaNs in aggregation
```
- Integer arrays can't hold `NaN` (float-only) — a common gotcha.

---

## 12. Sorting, Searching & Set Operations

```python
np.sort(a)                  # returns sorted copy
a.sort()                    # in-place sort
np.argsort(a)                # indices that would sort the array
np.sort(a, axis=0)           # sort per axis

np.searchsorted(a, 5)        # index to insert value keeping sorted order
np.where(a == 5)             # indices matching condition
np.nonzero(a)                # indices of nonzero elements
np.any(a > 5), np.all(a > 5)

np.unique(a)                  # unique sorted values
np.unique(a, return_counts=True)
np.intersect1d(a, b)
np.union1d(a, b)
np.setdiff1d(a, b)
np.isin(a, [1,2,3])
```

---

## 13. Iterating Over Arrays

```python
for x in a:                  # iterates over first axis (rows for 2D)
    print(x)

for x in np.nditer(a):       # flat iteration over every element
    print(x)

for idx, val in np.ndenumerate(a):   # index + value pairs
    print(idx, val)
```
- Iterating with Python loops defeats the purpose of vectorization — prefer vectorized ops when possible.

---

## 14. Structured / Record Arrays (less common but "even minor" territory)

```python
dt = np.dtype([('name', 'U10'), ('age', 'i4')])
people = np.array([('Ali', 25), ('Sara', 30)], dtype=dt)
people['name']       # array(['Ali','Sara'])
people['age'].mean()
```

---

## 15. Stacking, Tiling & Repeating

```python
np.tile(a, 3)             # repeat whole array 3 times
np.tile(a, (2,3))         # repeat in 2D pattern
np.repeat(a, 3)           # repeat each element 3 times
np.repeat(a, 3, axis=0)   # repeat rows
```

---

## 16. Random Sampling Details (np.random)

```python
rng.shuffle(a)                       # shuffle in-place
rng.permutation(a)                   # shuffled copy
rng.normal(loc=0, scale=1, size=10)  # gaussian
rng.uniform(0, 1, size=10)
rng.binomial(n=10, p=0.5, size=5)
rng.choice(a, size=3, replace=False, p=[weights])
```

---

## 17. Saving & Loading Arrays

```python
np.save('arr.npy', a)          # single array, binary
np.load('arr.npy')

np.savez('arrs.npz', x=a, y=b) # multiple arrays
data = np.load('arrs.npz')
data['x']

np.savetxt('a.csv', a, delimiter=',')
np.loadtxt('a.csv', delimiter=',')
np.genfromtxt('a.csv', delimiter=',', missing_values='NA')
```

---

## 18. Performance Notes / Best Practices

- Prefer vectorized ufuncs over Python `for` loops — orders of magnitude faster.
- Preallocate arrays (`np.zeros`, `np.empty`) instead of growing arrays in a loop.
- Use `np.float32` instead of `float64` when precision allows — saves memory.
- Avoid unnecessary `.copy()` calls; understand when views are created.
- Use `axis` parameter instead of manual loops for row/column-wise ops.
- `np.einsum` for advanced/optimized tensor contractions:
```python
np.einsum('ij,jk->ik', A, B)   # equivalent to A @ B, more flexible
```
- Check memory layout when performance matters: `arr.flags['C_CONTIGUOUS']`, `np.ascontiguousarray(a)`.

---

## 19. Common Pitfalls (things people forget)

1. `*` is element-wise, not matrix multiply — use `@` or `np.matmul`.
2. Broadcasting shape mismatches throw errors — always sanity check shapes.
3. Views vs copies — modifying a slice can silently modify the original.
4. `np.nan != np.nan` — NaN is never equal to itself; use `np.isnan()`.
5. Integer overflow: fixed-width int dtypes can silently wrap around (e.g., `int8` max is 127).
6. `axis=0` vs `axis=1` confusion — axis specifies which dimension is **collapsed**.
7. `np.array()` copies by default; `np.asarray()` does not copy if already an ndarray (useful for functions that shouldn't force copies).
8. Comparing floats with `==` is unreliable — use `np.isclose()` / `np.allclose()`.
9. `reshape` fails if total element count doesn't match — use `-1` to auto-infer one dimension.
10. Random seed only guarantees reproducibility if using the same RNG method (legacy `np.random.seed` vs `default_rng`).

---

## 20. Quick Cheat-Sheet Summary Table

| Task | Function |
|---|---|
| Create array | `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()` |
| Shape info | `.shape`, `.ndim`, `.size`, `.dtype` |
| Reshape | `.reshape()`, `.ravel()`, `.flatten()`, `.T` |
| Combine | `np.concatenate()`, `np.vstack()`, `np.hstack()`, `np.stack()` |
| Split | `np.split()`, `np.hsplit()`, `np.vsplit()` |
| Math | `+ - * / **`, `np.sqrt`, `np.exp`, `np.log` |
| Matrix mult | `@`, `np.matmul()`, `np.dot()` |
| Aggregate | `.sum()`, `.mean()`, `.std()`, `.min()`, `.max()`, `axis=` |
| Boolean mask | `a[a > x]`, `np.where()` |
| Sort/search | `np.sort()`, `np.argsort()`, `np.where()`, `np.unique()` |
| Missing data | `np.nan`, `np.isnan()`, `np.nan_to_num()` |
| Linear algebra | `np.linalg.inv()`, `.det()`, `.solve()`, `.eig()` |
| Save/load | `np.save()`, `np.load()`, `np.savetxt()` |
| Random | `np.random.default_rng()`, `.random()`, `.integers()`, `.normal()` |

---

### How to use this for revision
Go section by section and, for each one, try to write a 2-line code snippet from memory before checking the answer. The sections that map most directly to your ML/data-preprocessing work (broadcasting, boolean masking, axis-based aggregation, NaN handling) are worth drilling hardest, since those show up constantly in pandas + sklearn pipelines.