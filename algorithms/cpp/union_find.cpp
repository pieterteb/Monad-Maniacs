#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
struct UF {
    vi par, rank;
    size_t setc;
    UF(int n) : par(n), rank(n, 1), setc(n) {
        for (int i = 0; i < n; ++i) par[i] = i;
    };
};

/* Returns root of u. */
int uf_root(UF* uf, int u) {
    int r = uf->par[u];
    if (u == r) return r;
    return uf->par[u] = uf_root(uf, r);
}

void uf_union(UF* uf, int u, int v) {
    int ru = uf_root(uf, u);
    int rv = uf_root(uf, v);
    if (ru == rv) return;
    --uf->setc;
    if (uf->rank[ru] < uf->rank[rv]) {
        uf->par[ru] = rv;
    } else if (uf->rank[ru] > uf->rank[rv]) {
        uf->par[rv] = ru;
    } else {
        uf->par[rv] = ru;
        ++uf->rank[ru];
    }
}

bool uf_same_set(UF* uf, int u, int v) {
    return uf_root(uf, u) == uf_root(uf, v);
}

size_t uf_setc(UF* uf) {
    return uf->setc;
}
