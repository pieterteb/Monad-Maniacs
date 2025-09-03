#include <bits/stdc++.h>
using namespace std;
typedef vector<int> vi;
typedef deque<int> dqi;
typedef struct wedg { int v, w, r; } wedg;
typedef vector<vector<wedg>> wgr;

#define INFTY INT_MAX


void add_wedg(wgr& adj, int a, int b, int w) {
    adj[a].push_back({ b, w, static_cast<int>(adj[b].size()) });
    adj[b].push_back({ a, w, static_cast<int>(adj[a].size() - 1) });
}

void add_dwedg(wgr& adj, int a, int b, int w) {
    adj[a].push_back({ b, w, static_cast<int>(adj[b].size()) });
    adj[b].push_back({ a, 0, static_cast<int>(adj[a].size() - 1) });
}

int max_flow(wgr& adj, int src, int snk) {
    int flow = 0, N = adj.size(), minf, u, v, c;
    vi par(N), minc(N);
    vector<int*> fcap(N), rcap(N);
    while (true) {
        memset(&par[0], -1, N * sizeof(int));
        memset(&minc[0], 0, N * sizeof(int));
        dqi q; q.push_back(src);
        minc[src] = INFTY;
        while(!q.empty()) {
            u = q.front(); q.pop_front();
            for (auto& p : adj[u]) {
                v = p.v; c = p.w;
                if (par[v] == -1 && c > 0) {
                    par[v] = u; fcap[v] = &p.w; rcap[v] = &adj[v][p.r].w;
                    minc[v] = min(minc[u], c);
                    if (v == snk) goto path;
                    q.push_back(v);
                }
            }
        } break;
        path:
        minf = minc[snk];
        flow += minf;
        v = snk;
        while (v != src) {
            *fcap[v] -= minf;
            *rcap[v] += minf;
            v = par[v];
        }
    }
    return flow;
}
