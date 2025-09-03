#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;
typedef queue<int> qi;

void bfs(vvi& adj, int src) {
    qi q; q.push(src);
    vb seen(adj.size(), false); seen[src] = true;
    int u;
    while (!q.empty()) {
        u = q.front(); q.pop();
        for (auto v: adj[u]) {
            if (!seen[v]) {
                seen[v] = true;
                q.push(v);
            }
        }
    }
}
