#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;

bool bipartite(vvi& adj) {
    auto N = adj.size();
    vi st, c(N, 0);
    int u;
    for (auto i = 0; i < N; ++i) {
        if (c[i]) continue;
        st.push_back(i); c[i] = 1;
        while (st.size()) {
            u = st.back(); st.pop_back();
            for (auto v: adj[u]) {
                if (!c[v]) {
                    c[v] = 3 - c[u];
                    st.push_back(v);
                } else if (c[v] == c[u]) {
                    return false;
                }
            }
        }
    }
    return true;
}
