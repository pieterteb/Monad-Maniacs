#include <bits/stdc++.h>

using namespace std;

/* Computes maximum flow using an implementation of the Edmonds-Karp algorithm. */
template<typename T>
T max_flow(vector<vector<int>>& adj, vector<vector<T>> capacity, int source, int sink) {
    const T INFTY = numeric_limits<T>::max();
    T flow = 0;
    vector<int> parent(adj.size());
    while (true) {  
        fill(parent.begin(), parent.end(), -1);
        parent[source] = -2;
        queue<pair<int, T>> q;
        q.emplace(source, INFTY);
        bool found_path = false;
        T current_flow = INFTY;
        while(!q.empty() && !found_path) {
            int current_node = q.front().first;
            current_flow = q.front().second;
            q.pop();
            for (auto next_node : adj[current_node]) {
                if (parent[next_node] == -1 && capacity[current_node][next_node]) {
                    parent[next_node] = current_node;
                    current_flow = min(current_flow, capacity[current_node][next_node]);
                    if (next_node == sink) {
                        found_path = true;
                        break;
                    }
                    q.emplace(next_node, current_flow);
                }
            }
        }
        if (!found_path) break;
        flow += current_flow;
        int current_node = sink, previous_node;
        while (current_node != source) {
            previous_node = parent[current_node];
            capacity[previous_node][current_node] -= current_flow;
            capacity[current_node][previous_node] += current_flow;
            current_node = previous_node;
        }
    }
    return flow;
}
