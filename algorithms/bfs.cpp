#include <bits/stdc++.h>

using namespace std;

/* Breadth-first search implementation. */
void bfs(vector<vector<int>>& adj, int source) {
    vector<bool> visited(adj.size(), false);
    visited[source] = true;
    queue<int> q;
    q.push(source);
    while (!q.empty()) {
        source = q.front();
        q.pop();
        for (auto node : adj[source]) {
            if (!visited[node]) {
                visited[node] = true;
                q.push(node);
            }
        }
    }
}
