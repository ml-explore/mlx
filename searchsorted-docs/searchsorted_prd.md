# Product Requirements Document: `mlx.searchsorted`

**Document Version:** 1.0  
**Date:** November 19, 2025  
**Status:** Draft  
**GitHub Issue:** [#1255 - [Feature] searchsorted](https://github.com/ml-explore/mlx/issues/1255)

---

## 1. Executive Summary

This PRD outlines the requirements for implementing `mlx.searchsorted`, a new function in the MLX library that provides functionality equivalent to NumPy's `searchsorted`. This feature will enable efficient binary search operations on sorted arrays, a common requirement in scientific computing, machine learning, and data analysis workflows.

### Problem Statement
Currently, MLX lacks a native function to perform binary search operations on sorted arrays. Users need to find insertion indices for values in sorted sequences, which is a fundamental operation for:
- Binning and histogram operations
- Interpolation and lookup operations
- Custom sampling and distribution operations
- Data filtering and range queries

### Proposed Solution
Implement `mlx.searchsorted` to match NumPy's API, providing both left and right insertion semantics, with support for multi-dimensional arrays via an axis parameter.

---

## 2. Goals and Objectives

### Primary Goals
1. **API Compatibility:** Provide an API that is consistent with `numpy.searchsorted` to facilitate easy migration and familiar usage patterns
2. **Performance:** Leverage MLX's lazy evaluation and Metal backend for efficient GPU-accelerated binary search
3. **Multi-dimensional Support:** Enable searchsorted operations along any axis of multi-dimensional arrays
4. **Type Safety:** Ensure robust type handling and clear error messages

### Success Metrics
- Function passes all unit tests matching NumPy behavior
- Performance benchmarks show competitive or superior performance on GPU compared to NumPy on CPU
- Documentation is clear and comprehensive
- Zero regression in existing MLX functionality

### Non-Goals (Out of Scope)
- Custom comparator functions (use default numerical comparison only)
- Sorting validation (assume input is sorted as per NumPy behavior)
- Concurrent multi-axis searchsorted operations

---

## 3. User Stories

### User Story 1: Basic Binary Search
**As a** data scientist  
**I want to** find insertion indices for values in a sorted sequence  
**So that** I can efficiently bin and categorize data

**Acceptance Criteria:**
- Can search for a single scalar value in a 1D sorted array
- Can search for multiple values (1D array) in a 1D sorted array
- Returns correct insertion indices following left/right semantics

### User Story 2: Multi-dimensional Operations
**As a** ML engineer  
**I want to** perform searchsorted operations along specific axes of multi-dimensional arrays  
**So that** I can process batched data efficiently

**Acceptance Criteria:**
- Can specify axis parameter to search along any valid dimension
- Broadcasting rules are applied correctly
- Results maintain expected shape relationships

### User Story 3: Custom Insertion Semantics
**As a** numerical computing user  
**I want to** choose between left and right insertion semantics  
**So that** I can control handling of duplicate values in the sorted array

**Acceptance Criteria:**
- `side='left'` returns leftmost insertion index
- `side='right'` returns rightmost insertion index
- Behavior matches NumPy exactly for edge cases

---

## 4. Functional Requirements

### 4.1 Function Signature

```python
def searchsorted(
    a: array,
    v: Union[scalar, array],
    *,
    side: str = 'left',
    axis: Optional[int] = None,
    stream: Optional[Union[Stream, Device]] = None
) -> array:
    """
    Find indices where elements should be inserted to maintain order.
    
    Parameters
    ----------
    a : array
        Input array. If axis is None, a is flattened. Otherwise, a must be 
        sorted along the specified axis.
    v : scalar or array
        Values to insert into a.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location is returned.
        If 'right', return the last such index. Default is 'left'.
    axis : int, optional
        Axis along which to search. If None, a is flattened before searching.
        Default is None.
    stream : Stream or Device, optional
        Stream or device to schedule the operation on.
    
    Returns
    -------
    indices : array
        Array of insertion indices with the same shape as v.
    """
```

### 4.2 Input Validation

**Required Validations:**
1. `a` must be an MLX array
2. `v` must be a scalar or MLX array
3. `side` must be either `'left'` or `'right'`
4. If `axis` is specified:
   - Must be a valid integer within the range `[-a.ndim, a.ndim)`
   - Will be normalized to positive index
5. Types of `a` and `v` must be compatible for comparison
6. If `axis` is None, `a` will be flattened and `v` can be any shape
7. If `axis` is specified, broadcasting rules apply between `a` and `v`

**Error Messages:**
- Clear, actionable error messages for invalid inputs
- Follow MLX error message conventions

### 4.3 Core Behavior

#### 4.3.1 Single-dimensional Search (axis=None)
When `axis=None`:
1. Flatten array `a` to 1D
2. Perform binary search for each element in `v`
3. Return indices as if inserting into the flattened array
4. Output shape matches `v` shape

#### 4.3.2 Multi-dimensional Search (axis specified)
When `axis` is specified:
1. Perform binary search along the specified axis
2. Apply broadcasting rules between `a` and `v`
3. Return indices with appropriate shape based on broadcasting

#### 4.3.3 Side Semantics
- **`side='left'`**: For value `x`, return index `i` such that all elements in `a[:i]` are `< x` and all elements in `a[i:]` are `>= x`
- **`side='right'`**: For value `x`, return index `i` such that all elements in `a[:i]` are `<= x` and all elements in `a[i:]` are `> x`

### 4.4 Edge Cases

**Empty Arrays:**
- If `a` is empty and `axis=None`, return array of zeros with shape of `v`
- If searching along an axis of length 0, return array of zeros appropriately shaped

**Single Element:**
- Correctly handle arrays with single elements
- Return 0 or 1 based on comparison and side

**All Duplicates:**
- When all elements in `a` equal the search value:
  - `side='left'` returns 0
  - `side='right'` returns `len(a)` (or size along axis)

**Out of Range Values:**
- Values less than all elements in `a` return 0
- Values greater than all elements in `a` return `len(a)` (or size along axis)

**Type Promotion:**
- Follow MLX type promotion rules when comparing `a` and `v`
- Support mixed integer/float comparisons

---

## 5. Technical Requirements

### 5.1 Implementation Approach

**Python API Layer** (`python/mlx/array.py` or `python/mlx/_funcs.py`):
- Input validation and type checking
- Axis normalization
- Dispatch to C++ backend

**C++ Core** (`mlx/ops.h`, `mlx/ops.cpp`):
- Define `searchsorted` operation
- Handle lazy evaluation graph construction
- Define primitive operation

**Primitive Implementation** (`mlx/backend/metal/` or `mlx/backend/common/`):
- Efficient binary search algorithm
- Metal kernel for GPU acceleration
- Fallback CPU implementation

### 5.2 Algorithm

**Binary Search Algorithm:**
- Use standard binary search with O(log n) complexity per search
- For vectorized searches, parallelize across search values
- Consider optimizations for:
  - Batch processing
  - Cache-friendly memory access patterns
  - GPU thread utilization

**Metal Kernel Considerations:**
- Efficient thread mapping for parallel searches
- Minimize divergence in binary search paths
- Optimize memory access patterns

### 5.3 Performance Requirements

**Target Performance:**
- Single search: O(log n) time complexity
- Vectorized search: O(m * log n) where m = number of values, n = array size
- GPU acceleration for large arrays (when applicable)
- Competitive with or faster than NumPy on equivalent hardware for relevant problem sizes

**Memory Requirements:**
- O(1) additional memory beyond output array
- Lazy evaluation support (no immediate materialization)

### 5.4 Dependencies

**Internal MLX Dependencies:**
- Core array operations
- Type promotion system
- Stream/device management
- Lazy evaluation graph

**No New External Dependencies Required**

---

## 6. Testing Requirements

### 6.1 Unit Tests

**Basic Functionality:**
```python
# Test 1D arrays with scalars
def test_searchsorted_1d_scalar():
    a = mx.array([1, 2, 3, 4, 5])
    assert mx.searchsorted(a, 3) == 2
    assert mx.searchsorted(a, 3, side='right') == 3

# Test 1D arrays with array values
def test_searchsorted_1d_array():
    a = mx.array([1, 2, 3, 4, 5])
    v = mx.array([0, 3, 6])
    expected = mx.array([0, 2, 5])
    assert mx.array_equal(mx.searchsorted(a, v), expected)

# Test with axis parameter
def test_searchsorted_axis():
    a = mx.array([[1, 2, 3], [4, 5, 6]])
    v = mx.array([2, 5])
    result = mx.searchsorted(a, v, axis=1)
    # Verify shape and values

# Test edge cases
def test_searchsorted_edge_cases():
    # Empty array
    # Single element
    # All duplicates
    # Out of range values
```

**NumPy Compatibility Tests:**
- Compare results against NumPy for identical inputs
- Test all parameter combinations
- Verify broadcasting behavior matches NumPy

**Type Tests:**
- Integer arrays
- Float arrays
- Mixed precision
- Type promotion

**Error Handling:**
- Invalid side parameter
- Invalid axis parameter
- Type incompatibility
- Invalid array inputs

### 6.2 Integration Tests

- Test with other MLX operations in computation graphs
- Verify lazy evaluation behavior
- Test with different stream/device configurations

### 6.3 Performance Tests

- Benchmark against NumPy
- Profile GPU vs CPU performance
- Test scalability with array size
- Memory usage validation

---

## 7. Documentation Requirements

### 7.1 API Documentation

**Docstring Requirements:**
- Complete parameter descriptions
- Return value description
- Type hints
- Examples section with 3-5 examples covering common use cases
- Notes section explaining behavior, assumptions, and edge cases
- See Also section linking to related functions

**Example Documentation:**
```python
Examples
--------
Find insertion indices in a 1D array:

>>> a = mx.array([1, 3, 5, 7, 9])
>>> mx.searchsorted(a, 5)
array(2)

Search for multiple values:

>>> v = mx.array([0, 4, 10])
>>> mx.searchsorted(a, v)
array([0, 2, 5])

Use right-side insertion:

>>> a = mx.array([1, 2, 2, 2, 5])
>>> mx.searchsorted(a, 2, side='left')
array(1)
>>> mx.searchsorted(a, 2, side='right')
array(4)

Search along a specific axis:

>>> a = mx.array([[1, 2, 3], [4, 5, 6]])
>>> mx.searchsorted(a, 3, axis=1)
array([2, 0])
```

### 7.2 User Guide

Add section to MLX documentation:
- Overview of searchsorted functionality
- Common use cases and patterns
- Performance tips
- Comparison with NumPy

### 7.3 Release Notes

Document in release notes when shipped:
- Brief description of new feature
- Link to documentation
- Acknowledgment of original issue/contributor

---

## 8. User Experience

### 8.1 Learning Curve

**For NumPy Users:**
- Should be immediately familiar
- Drop-in replacement in most cases
- Migration guide for any differences

**For New Users:**
- Clear, intuitive API
- Comprehensive examples
- Good error messages

### 8.2 Error Messages

Design clear, actionable error messages:

```python
# Example error messages:
"searchsorted: side must be 'left' or 'right', got '{side}'"
"searchsorted: axis {axis} is out of bounds for array of dimension {ndim}"
"searchsorted: arrays a and v must have compatible types for comparison"
```

---

## 9. Risks and Mitigation

### 9.1 Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Performance doesn't meet expectations | Medium | Low | Benchmark early, optimize iteratively |
| Edge case behavior differs from NumPy | High | Medium | Comprehensive testing against NumPy |
| GPU implementation complexity | Medium | Medium | Start with CPU, optimize GPU later |
| Type promotion edge cases | Medium | Low | Leverage existing MLX type system |

### 9.2 Breaking Changes

**None Expected** - This is a new feature addition with no impact on existing functionality.

---

## 10. Implementation Phases

### Phase 1: Core Implementation (Week 1-2)
**Deliverables:**
- [ ] Unit tests for basic functionality
- [ ] Python API implementation
- [ ] C++ operation definition
- [ ] Basic CPU primitive implementation

**Success Criteria:**
- All basic 1D tests pass
- NumPy compatibility for simple cases

### Phase 2: Multi-dimensional Support (Week 2-3)
**Deliverables:**
- [ ] Extended unit tests
- [ ] NumPy compatibility tests
- [ ] Axis parameter support
- [ ] Broadcasting implementation

**Success Criteria:**
- All axis-based tests pass
- Broadcasting works correctly
- 100% NumPy compatibility

### Phase 3: Optimization (Week 3-4)
**Deliverables:**
- [ ] Benchmarking suite
- [ ] Metal GPU kernel implementation
- [ ] Performance optimizations
- [ ] Performance documentation

**Success Criteria:**
- GPU acceleration working
- Competitive performance metrics
- No performance regressions

### Phase 4: Documentation and Release (Week 4)
**Deliverables:**
- [ ] Complete API documentation
- [ ] User guide updates
- [ ] Examples and tutorials
- [ ] Release notes

**Success Criteria:**
- Documentation review complete
- Examples verified
- Ready for PR submission

---

## 11. Success Criteria

The implementation will be considered complete and successful when:

1. ✅ **Functionality:** All unit tests pass with 100% NumPy compatibility
2. ✅ **Performance:** Meets or exceeds performance targets on both CPU and GPU
3. ✅ **Documentation:** Complete, clear, and approved documentation
4. ✅ **Code Quality:** Passes code review and meets MLX coding standards
5. ✅ **Integration:** Successfully merges into main branch without breaking existing tests
6. ✅ **User Validation:** Positive feedback from initial users/reviewers

---

## 12. Open Questions

1. **Q:** Should we support sorter parameter (like NumPy) for indirect sorting?  
   **A:** Not in initial implementation - can be added later if needed

2. **Q:** How should we handle NaN values in arrays?  
   **A:** Follow NumPy behavior - NaN comparison semantics

3. **Q:** Should we validate that input array is actually sorted?  
   **A:** No - match NumPy behavior (undefined for unsorted input)

4. **Q:** Any special handling for complex numbers?  
   **A:** Initially only support real-valued numeric types

---

## 13. References

- **GitHub Issue:** https://github.com/ml-explore/mlx/issues/1255
- **NumPy Documentation:** https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
- **MLX Documentation:** https://ml-explore.github.io/mlx/
- **Related Work:** Implementation guides in `/searchsorted-docs/onboarding_guide.md`

---

## 14. Appendix

### A. NumPy API Reference

```python
numpy.searchsorted(a, v, side='left', sorter=None)
```

**Key Differences from MLX Proposal:**
- NumPy has `sorter` parameter (not in initial MLX implementation)
- MLX adds explicit `stream` parameter for device control
- MLX adds explicit `axis` parameter (NumPy uses separate `axis` in some contexts)

### B. Example Use Cases

**Use Case 1: Histogram Binning**
```python
import mlx.core as mx

# Define bin edges
bins = mx.array([0, 10, 20, 30, 40, 50])
# Data to bin
data = mx.array([5, 15, 25, 35, 45, 3, 47])
# Find bin indices
indices = mx.searchsorted(bins, data)
```

**Use Case 2: Interpolation**
```python
# Find insertion points for interpolation
x = mx.array([0, 1, 2, 3, 4])
x_new = mx.array([0.5, 1.5, 2.5])
indices = mx.searchsorted(x, x_new)
# Use indices for linear interpolation
```

**Use Case 3: Range Queries**
```python
# Find all elements in range [low, high)
sorted_data = mx.array([1, 3, 5, 7, 9, 11, 13])
low, high = 5, 11
left_idx = mx.searchsorted(sorted_data, low, side='left')
right_idx = mx.searchsorted(sorted_data, high, side='left')
range_elements = sorted_data[left_idx:right_idx]
```

---

**Document Prepared By:** Development Team  
**Review Status:** Pending Review  
**Next Review Date:** TBD
