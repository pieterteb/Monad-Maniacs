#include <bits/stdc++.h>

using namespace std;

/* Iterative depth-first search implementation. */
void dfs(vector<vector<int>>& adj, int source) {
    vector<bool> visited(adj.size(), false);
    visited[source] = true;
    stack<int> st;
    st.push(source);
    while (!st.empty()) {
        source = st.top();
        st.pop();
        for (auto node : adj[source]) {
            if (!visited[node]) {
                visited[node] = true;
                st.push(node);
            }
        }
    }
}

/* Recursive depth-first search implementation. */
void dfs(vector<vector<int>>& adj, int source, vector<bool>& visited) {
    for (auto node : adj[source]) {
        if (!visited[node]) {
            visited[node] = true;
            dfs(adj, node, visited);
        }
    }
}
