#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;
typedef queue<int> qi;

bool bfs_srch(vvi& adj, int src, int dest) {
    qi q; q.push(src);
    vb seen(adj.size(), false); seen[src] = true;
    int u;
    while (!q.empty()) {
        u = q.front(); q.pop();
        for (auto v: adj[u]) {
            if (v == dest) return true;
            if (!seen[v]) {
                seen[v] = true;
                q.push(v);
            }
        }
    }
    return false;
}
