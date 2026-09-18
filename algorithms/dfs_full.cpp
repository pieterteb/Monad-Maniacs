#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;

void dfs_full(vvi& adj) {
    auto N = adj.size();
    vi st;
    vb seen(N, false);
    int u;
    for (auto i = 0; i < N; ++i) {
        if (seen[i]) continue;
        seen[i] = true;
        st.push_back(i);
        while (!st.empty()) {
            u = st.back(); st.pop_back();
            for (auto v: adj[u]) {
                if (!seen[v]) {
                    seen[v] = true;
                    st.push_back(v);
                }
            }
        }
    }
}
