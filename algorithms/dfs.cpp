#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;

void dfs(vvi& adj, int src) {
    vi st; st.push_back(src);
    vb seen(adj.size(), false); seen[src] = true;
    int u;
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
