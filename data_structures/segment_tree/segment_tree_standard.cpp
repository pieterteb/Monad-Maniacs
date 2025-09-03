#include <bits/stdc++.h>
using namespace std;

/* Segment tree struct. */
template<typename T>
struct SegmentTree {
    int element_count;              // Number of elements.
    function<T(T, T)> operation;    // Associative operation.
    const T identity;               // Value such that operation(a, identity) == a == operation(identity, a) for all valid a.
    vector<T> tree;                 // Stores the binary tree.

    /* Build a segment tree with n elements and initializes them to id, with operation op and identity i. */
    SegmentTree(int n, function<T(T, T)> op, T id) : element_count(n), operation(op), identity(id), tree(2 * n, id) {}

    /* Build a segment tree from arr with operation op and identity id. */
    SegmentTree(vector<T>& arr, function<T(T, T)> op, T id) : element_count(arr.size()), operation(op), identity(id), tree(2 * arr.size()) {
        copy(arr.begin(), arr.end(), tree.begin() + element_count);
        for (int i = element_count - 1; i > 0; --i)
            tree[i] = operation(tree[2 * i], tree[2 * i + 1]);
    }
};

/* Returns cumulative result of applying segtree.operation on [left, right). Returns segtree.identity if the interval is empty. */
template<typename T>
T segment_tree_query(SegmentTree<T>& segtree, unsigned left, unsigned right) {
    T result = segtree.identity;
    left += segtree.element_count;
    right += segtree.element_count;
    while (left < right) {
        if (left & 1) result = segtree.operation(result, segtree.tree[left++]);
        if (right & 1) result = segtree.operation(result, segtree.tree[--right]);
        left >>= 1;
        right >>= 1;
    }
    return result;
}

/* Assigns value to element at index. */
template<typename T>
void segment_tree_assign(SegmentTree<T>& segtree, unsigned index, T value) {
    index += segtree.element_count;
    segtree.tree[index] = value;
    while (index > 1) {
        segtree.tree[index >> 1] = segtree.operation(segtree.tree[index], segtree.tree[index ^ 1]);
        index >>= 1;
    }
}
