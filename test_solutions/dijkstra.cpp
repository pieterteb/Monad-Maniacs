#include <bits/stdc++.h>

using namespace std;


vector<int> dijkstra(vector<vector<pair<int, int>>>& adj, int src) {
    vector<int> dist(adj.size(), INT_MAX); // distance
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q; // { distance, node }
    dist[src] = 0;
    q.emplace(0, src);
    while (!q.empty()) {
        auto [du, u] = q.top(); q.pop();
        if (du != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                q.emplace(dist[v], v);
            }
        }
    }
    return dist;
}


vector<vector<pair<int, int>>> graph = {
    {{1, 0}},
    {{2, 5}, {3, 8}, {5, 7}, {6, 10}, {0, 1}},
    {{3, 1}, {5, 3}},
    {{4, 6}},
    {},
    {{4, 4}, {6, 2}, {7, 7}},
    {{7, 3}},
    {{3, 4}, {4, 5}}
};

int main() {
    while (true) {
        int a;
        cin >> a;

        vector<int> dist = dijkstra(graph, a);
        for (auto d : dist) cout << d << " ";
        cout << endl;
    }
}