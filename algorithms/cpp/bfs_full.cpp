#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<bool> vb;
typedef queue<int> qi;

void bfs_full(vvi& adj) {
    auto N = adj.size();
    qi q;
    vb seen(N, false);
    int u;
    for (auto i = 0; i < N; ++i) {
        if (seen[i]) continue;
        seen[i] = true;
        q.push(i);
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
}
