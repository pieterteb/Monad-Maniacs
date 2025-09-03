#include <bits/stdc++.h>

using namespace std;


/* Segment Tree data structure. */
template<typename T>
struct SegTree {
    int elementc;
    vector<T> tree; // Root has index 1; the 0th element is unused.
    
    // CHANGE THIS SEGMENT
    const T id = -INFINITY;
    T operation(T a, T b) {
        return max(a, b);
    }
    // UNTIL HERE

    /* Build a segment tree with n elements and initializes them to id. */
    SegTree(int n) : elementc(n), tree(2 * n, id) {}

    /* Build a segment tree from arr. */
    SegTree(vector<T>& arr) : elementc(arr.size()), tree(2 * arr.size(), id) {
        copy(arr.begin(), arr.end(), tree.begin() + elementc);      // Copy the array values to the bottom of the tree
        for (int i = elementc - 1; i > 0; --i)
            tree[i] = operation(tree[2 * i], tree[2 * i + 1]);
    }

    /* Returns cumulative result of applying segtree.operation on [left, right). Returns segtree.identity if the interval is empty. */
    T query(unsigned left, unsigned right) {
        T result = id;
        left += elementc;
        right += elementc;
        while (left < right) {
            if (left & 1) result = operation(result, tree[left++]);
            if (right & 1) result = operation(result, tree[--right]);
            left >>= 1; // Dividing by 2
            right >>= 1;
        }
        return result;
    }

    /* Assigns value to element at index. */
    void assign(unsigned index, T value) {
        index += elementc;
        tree[index] = value;
        while (index > 1) {
            tree[index >> 1] = operation(tree[index], tree[index ^ 1]);
            index >>= 1;
        }
    }
};


int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    }
    SegTree st(arr);

    

    return 0;
}
