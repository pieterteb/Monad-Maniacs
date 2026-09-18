#include <bits/stdc++.h>

using namespace std;

#define ID 1

typedef vector<int> vi;
typedef struct segtr { int N; vi tr; int (*op)(int, int); segtr(int n, int (*o)(int, int)) : N(n), tr(2 * n, ID), op(o) {} } segtr;

int segtr_op(int a, int b) {
    return a * b;
}

segtr* segtr_build(vi& arr, int (*op)(int, int)) {
    segtr* tr = new segtr(arr.size(), op);
    copy(arr.begin(), arr.end(), tr->tr.begin() + tr->N);
    for (int i = tr->N - 1; i > 0; --i)
        tr->tr[i] = tr->op(tr->tr[2 * i], tr->tr[2 * i + 1]);
    return tr;
}

segtr* segtr_build(int l, int (*op)(int, int)) {
    segtr* tr = new segtr(l, op);
    return tr;
}

/* Gets result in the interval [l, r). */
int segtr_query(segtr* tr, unsigned int l, unsigned int r) {
    int res = ID;
    l += tr->N;
    r += tr->N;
    while (l < r) {
        if (l & 1) res = tr->op(res, tr->tr[l++]);
        if (r & 1) res = tr->op(res, tr->tr[--r]);
        l >>= 1;
        r >>= 1;
    }
    return res;
}

void segtr_modify(segtr* tr, unsigned int i, int val) {
    i += tr->N;
    tr->tr[i] = val;
    while (i > 1) {
        tr->tr[i / 2] = tr->op(tr->tr[i], tr->tr[i ^ 1]);
        i >>= 1;
    }
}
