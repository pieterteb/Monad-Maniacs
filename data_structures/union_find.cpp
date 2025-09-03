#include <bits/stdc++.h>

using namespace std;

/* Union Find data structure. */
struct UF {
    vector<int> par, rank;
    int setc;
    UF(int n) : par(n), rank(n, 0), setc(n) {
        for (int i = 0; i < n; ++i) par[i] = i;
    };

    int root(int u) {
        int r = par[u];
        if (u == r) return r;
        return par[u] = root(r);
    }

    void unite(int u, int v) {
        int ru = root(u);
        int rv = root(v);
        if (ru == rv) return;
        --setc;
        if (rank[ru] < rank[rv]) {
            par[ru] = rv;
        } else if (rank[ru] > rank[rv]) {
            par[rv] = ru;
        } else {
            par[rv] = ru;
            ++rank[ru];
        }
    }

    bool same_set(int u, int v) {
        return root(u) == root(v);
    }
};
