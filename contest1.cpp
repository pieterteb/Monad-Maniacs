#include <bits/stdc++.h>
using namespace std;

/* 128 bit integer input */
__int128 strToi128(string s) {
    __int128 res = 0;
    bool neg = false;
    int i = 0;
    if (s[i] == '-') {
        neg = true;
        ++i;
    }
    for (; i < s.size(); ++i)
        res = res * 10 + (s[i] - '0');
    return neg ? -res : res;
}

/* Graph data structures. */

/* Unweighted adjacency list graph. */
typedef vector<int> vi;
typedef vector<vi> vvi;

void add_edge(vvi& adj, int a, int b) {
    adj[a].push_back(b);
}

/* Weighted edgelist graph. */
struct edg {
    int u, v, w;
};
typedef vector<edg> edgl;

void add_edge(edgl& edgl, int u, int v, int w) {
    edgl.push_back({ u, v, w });
}

/* Max flow graph. */
struct wedg {
    int v, w, r;
};
typedef vector<vector<wedg>> wgr;

void add_wedg(wgr& adj, int a, int b, int w) {
    adj[a].push_back({ b, w, static_cast<int>(adj[b].size()) });
    adj[b].push_back({ a, w, static_cast<int>(adj[a].size() - 1) });
}

void add_dwedg(wgr& adj, int a, int b, int w) {
    adj[a].push_back({ b, w, static_cast<int>(adj[b].size()) });
    adj[b].push_back({ a, 0, static_cast<int>(adj[a].size() - 1) });
}

/* Binary search. */
typedef vector<int> vi;

int binsrch(vi& a, int t) {
    int l = 0, r = a.size() - 1, m;
    while (l <= r) {
        m = l + (r - l) / 2;
        if (a[m] == t)
            return m;
        else if (a[m] < t)
            l = m + 1;
        else
            r = m - 1;
    }
    return -1;
}

/* Depth first search. */
typedef vector<bool> vb;
typedef vector<int> vi;
typedef vector<vi> vvi;

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

/* Breadth first search. */
typedef vector<bool> vb;
typedef queue<int> qi;
typedef vector<int> vi;
typedef vector<vi> vvi;

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

/* Check if bipartite */
typedef vector<bool> vb;
typedef vector<int> vi;
typedef vector<vi> vvi;

bool bipartite(vvi& adj) {
    auto N = adj.size();
    vi st, c(N, 0);
    int u;
    for (auto i = 0; i < N; ++i) {
        if (c[i]) continue;
        st.push_back(i); c[i] = 1;
        while (st.size()) {
            u = st.back(); st.pop_back();
            for (auto v: adj[u]) {
                if (!c[v]) {
                    c[v] = 3 - c[u];
                    st.push_back(v);
                } else if (c[v] == c[u]) {
                    return false;
                }
            }
        }
    }
    return true;
}

/* Union find datastructure. */
typedef vector<int> vi;

struct UF {
    vi par, rank;
    size_t setc;
    UF(int n) : par(n), rank(n, 1), setc(n) {
        for (int i = 0; i < n; ++i) par[i] = i;
    };
};

int uf_root(UF* uf, int u) {
    int r = uf->par[u];
    if (u == r) return r;
    return uf->par[u] = uf_root(uf, r);
}

void uf_union(UF* uf, int u, int v) {
    int ru = uf_root(uf, u);
    int rv = uf_root(uf, v);
    if (ru == rv) return;
    --uf->setc;
    if (uf->rank[ru] < uf->rank[rv]) {
        uf->par[ru] = rv;
    } else if (uf->rank[ru] > uf->rank[rv]) {
        uf->par[rv] = ru;
    } else {
        uf->par[rv] = ru;
        ++uf->rank[ru];
    }
}

bool uf_same_set(UF* uf, int u, int v) {
    return uf_root(uf, u) == uf_root(uf, v);
}

size_t uf_setc(UF* uf) {
    return uf->setc;
}

/* Kruskal minimal spanning tree. Needs weighted edge list and union find. */
int kruskal(edgl& edgel, int n) {
    sort(edgel.begin(), edgel.end(), [](edg& a, edg& b) { return a.w < b.w; });
    int weight = 0;
    UF* uf = new UF(n);
    for (auto& e : edgel) {
        if (!uf_same_set(uf, e.u, e.v)) {
            uf_union(uf, e.u, e.v);
            weight += e.w;
        }
    }
    delete uf;
    return weight;
}

/* Tarjan SCC. */
void SCC(int u, vector<vector<int>> &adj, int &curnum, int &compc, vector<int> &num, vector<int> &lowln, vector<int> &comp, vector<int> &stack) {
    stack.push_back(u);
    lowln[u] = num[u] = curnum++;

    for (int v : adj[u]) {
        if (num[v] == -1) {
            SCC(v, adj, curnum, compc, num, lowln, comp, stack);
            lowln[u] = min(lowln[u], lowln[v]);
        } else if (comp[v] == -1) lowln[u] = min(lowln[u], lowln[v]);
    }

    if (num[u] == lowln[u]) {
        for (int v = -1; v != u;) {
            v = stack.back(); stack.pop_back();
            comp[v] = compc;
        }
        compc++;
    }
}

/* Edmonds karp max flow. Needs max flow graph. */
#define INFTY INT_MAX
typedef queue<int> qi;
typedef vector<int> vi;

int max_flow(wgr& adj, int src, int snk) {
    int flow = 0, N = adj.size(), minf, u, v, c;
    vi par(N), minc(N);
    vector<int*> fcap(N), rcap(N);
    while (true) {
        memset(&par[0], -1, N * sizeof(int));
        memset(&minc[0], 0, N * sizeof(int));
        qi q; q.push(src);
        minc[src] = INFTY;
        while(!q.empty()) {
            u = q.front(); q.pop();
            for (auto& p : adj[u]) {
                v = p.v; c = p.w;
                if (par[v] == -1 && c > 0) {
                    par[v] = u; fcap[v] = &p.w; rcap[v] = &adj[v][p.r].w;
                    minc[v] = min(minc[u], c);
                    if (v == snk) goto path;
                    q.push(v);
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

/* Segment tree. */
#define ID 1
typedef vector<int> vi;
struct segtr {
    int N;
    vi tr;
    int (*op)(int, int);
    segtr(int n, int (*o)(int, int)) : N(n), tr(2 * n, ID), op(o) {}
};

int segtr_op(int a, int b) {
    return a * b;
}

segtr* segtr_build(vi& arr, int (*op)(int, int)) {
    segtr* tr = new segtr(arr.size(), op);
    copy(arr.begin(), arr.end(), tr->tr.begin() + tr->N);
    for (int i = tr->N - 1; i > 0; --i)
        tr->tr[i] = tr->op(tr->tr[2 * i], tr->tr[2 * i + 1]);
    return tr;
}

segtr* segtr_build(int l, int (*op)(int, int)) {
    segtr* tr = new segtr(l, op);
    return tr;
}

/* Gets result in the interval [l, r). */
int segtr_query(segtr* tr, unsigned int l, unsigned int r) {
    int res = ID;
    l += tr->N;
    r += tr->N;
    while (l < r) {
        if (l & 1) res = tr->op(res, tr->tr[l++]);
        if (r & 1) res = tr->op(res, tr->tr[--r]);
        l >>= 1;
        r >>= 1;
    }
    return res;
}

void segtr_modify(segtr* tr, unsigned int i, int val) {
    i += tr->N;
    tr->tr[i] = val;
    while (i > 1) {
        tr->tr[i / 2] = tr->op(tr->tr[i], tr->tr[i ^ 1]);
        i >>= 1;
    }
}

/* Knuth Morris Pratt algorithm. */
typedef vector<int> vi;

int KMP(const string &s, const string &t) {
    int m = t.size();
    vi pi(m + 1);
    pi[0] = 0;
    if (m) pi[1] = 0;
    for (int i = 2; i <= m; ++i) {
        for (int j = pi[i - 1]; ; j = pi[j]) {
            if (t[j] == t[i - 1]) {
                pi[i] = j + 1;
                break;
            }
            if (j == 0) {
                pi[i] = 0;
                break;
            }
        }
    }
    int count = 0;
    int n = s.size();
    for (int i = 0, j = 0; i < n; ) {
        if (s[i] == t[j]) {
            ++i; ++j;
            if (j == m) {
                ++count;
                j = pi[j];
            }
        }
        else if (j > 0) j = pi[j];
        else ++i;
    }
    return count;
}

/* Tries. */
typedef struct Trie {
    Trie* child[26];
    bool end;
    bool has_child;
    Trie() : child{}, end(false), has_child(false) {};
} Trie;

void trie_insert(Trie* root, string& s) {
    for (auto c : s) {
        root->has_child = true;
        c -= 'a';
        if (!root->child[c])
            root->child[c] = new Trie();
        root = root->child[c];
    }
    root->end = true;
}

void trie_delete(Trie* root) {
    for (Trie* child : root->child)
        if (child)
            trie_delete(child);
    delete root;
}

bool trie_contains(Trie* root, string& s) {
    for (auto c : s) {
        c -= 'a';
        if (!root->child[c])
            return false;   
        root = root->child[c];
    }
    return root->end;
}

/* Suffix array. */
struct suffix_array {
    struct entry {
        pair<int, int> nr;
        int p;
        bool operator <(const entry &other) const {
            return nr < other.nr;
        }
    };
    string s;
    int n;
    vector<vector<int> > P;
    vector<entry> L;
    vi idx;
    
    suffix_array(string _s) : s(_s), n(s.size()) {
        L = vector<entry>(n);
        P.push_back(vi(n));
        idx = vi(n);
        for (int i = 0; i < n; i++) {
            P[0][i] = s[i];
        }
        for (int stp = 1, cnt = 1; (cnt >> 1) < n; stp++, cnt <<= 1) {
            P.push_back(vi(n));
            for (int i = 0; i < n; i++) {
                L[i].p = i;
                L[i].nr = make_pair(P[stp - 1][i], i + cnt < n ? P[stp - 1][i + cnt] : -1);
            }
            sort(L.begin(), L.end());
            for (int i = 0; i < n; i++) {
                if (i > 0 && L[i].nr == L[i - 1].nr) {
                    P[stp][L[i].p] = P[stp][L[i - 1].p];
                } else {
                    P[stp][L[i].p] = i;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            idx[P[P.size() - 1][i]] = i;
        }
    }

    /* Longest common prefix. */
    int lcp(int x, int y) {
        int res = 0;
        if (x == y) return n - x;
        for (int k = P.size() - 1; k >= 0 && x < n && y < n; k--) {
            if (P[k][x] == P[k][y]) {
                x += 1 << k;
                y += 1 << k;
                res += 1 << k;
            }
        }
        return res;
    }
};

/* Miller primality. */
typedef unsigned long long ull;

ull multmod(ull a, ull b, ull m) {
    return (unsigned __int128)a * b % m;
}

ull powmod(ull a, ull b, ull m) {
    ull res = 1;
    while(b) {
        if (b & 1)
            res = multmod(res, a, m);
        a = multmod(a, a, m);
        b >>= 1;
    }
    return res;
}

int bit_length(long long n) {
    int l = 0;
    while(n) {
        n >>= 1;
        ++l;
    }
    return l;
}

bool prime(ull n) {
    if (n < 3 || !(n & 1))
        return n == 2;
    ull N = n - 1;
    int s = bit_length((N & -N)) - 1;
    ull d = N >> s;
    int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for(auto p : primes) {
        if (p >= n) break;
        ull x = powmod(p, d, n);
        ull y;
        for(size_t j = 0; j < s; ++j) {
            y = powmod(x, 2, n);
            if (y == 1 && x != 1 && x != n - 1)
                return false;
            x = y;
        }
        if (y != 1)
            return false;
    }
    return true;
}

/* Gcd Euclidean algorithm. */
int gcd(int a, int b) {
    int temp;
    while (b != 0) {
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/* Extended gcd. */
std::tuple<int, int, int> extgcd(int a, int b) {
    int x0 = 1, x1 = 0, y0 = 0, y1 = 1;
    int q, r, tempx, tempy;
    while (b != 0) {
        q = a / b;
        r = a % b;
        a = b;
        b = r;
        tempx = x0 - q * x1;
        tempy = y0 - q * y1;
        x0 = x1;
        x1 = tempx;
        y0 = y1;
        y1 = tempy;
    }
    return {a, x0, y0};
}

/* Toposort khan. */
vector<int> toposort(int n, vector<vector<int>> &adj) {
    vector<int> res, stack, p(n);
    
    for (int u = 0; u < n; u++) for (int child : adj[u]) p[child]++;
    for (int u = 0; u < n; u++) if (p[u] == 0) stack.push_back(u);

    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        res.push_back(u);

        for (int v : adj[u]) {
            p[v]--;
            if (p[v] == 0) stack.push_back(v);
        }
    }

    if (res.size() < n) return {-1};
    return res;
}

/* Geometry. */
#define eps 0.00000001
#define oo INT_MAX
typedef double coord;
struct pt{
coord x,y;
    pt():x(0),y(0){};
    pt(coord _x,coord _y):x(_x),y(_y){};
    pt operator+(const pt& p) {
    return pt(x+p.x,y+p.y);
    }
    pt operator-(const pt& p) {
    return pt(x-p.x,y-p.y);
    }
    coord operator*(const pt& p) {
    return x*p.x+y*p.y;
    }
    pt operator*(const coord c) {
        return pt(c*x, c*y);
    }
};
double len(pt p) { return sqrt(double(p*p)); }

pt rotate(pt p, double phi) {
    return pt(p.x*cos(phi)-p.y*sin(phi),
    p.x*sin(phi)+p.y*cos(phi));
}
pt perp(pt p){
    pt pp(p.y, -p.x);
    return pp;
}
pt closestpt(pt a0, pt a1, pt p) {
    if ((a1 - a0) * (p - a1) > 0) return a1;
    if ((a0 - a1) * (p - a0) > 0) return a0;
    pt d=a1-a0;
    return a0+d*((d*(p-a0))/(d*d));
}
double cross(pt a, pt b) {
    return a.x*b.y - a.y*b.x;
}
int ccw(pt p0, pt p1, pt p2) {
    coord d1 =(p1.x-p0.x)*(p2.y-p0.y);
    coord d2 =(p2.x-p0.x)*(p1.y-p0.y);
    return (d1-d2>eps)-(d2-d1>eps);
}
int isPointOnSegment(pt p, pt a0, pt a1) {
    if(ccw(a0,a1,p)) return 0;
    coord cx = (p.x-a0.x)*(p.x-a1.x);
    coord cy = (p.y-a0.y)*(p.y-a1.y);
    if(cx > eps || cy > eps) return 0;
    if(cx < -eps || cy < -eps) return 2;
    return 1;
}
int isSegmentIntersect(pt a0, pt a1, pt b0, pt b1) {
    int c1=ccw(a0,a1,b0);
    int c2=ccw(a0,a1,b1);
    int c3=ccw(b0,b1,a0);
    int c4=ccw(b0,b1,a1);
    if(c1*c2>0 || c3*c4>0) return 0;
    if(!c1 && !c2 && !c3 && !c4) {
        c1=isPointOnSegment(a0,b0,b1);
        c2=isPointOnSegment(a1,b0,b1);
        c3=isPointOnSegment(b0,a0,a1);
        c4=isPointOnSegment(b1,a0,a1);
        if(c1 && c2 && c3 && c4) return 1+(a0.x!=a1.x || a0.y!=a1.y);
        if (c1 + c2 + c3 + c4 == 0) return 0;
        return 3 - max({c1,c2,c3,c4});
    }
    return 1+(!c1 || !c2 || !c3 || !c4);
}
pt lineIntersect(pt a0, pt a1, pt b0, pt b1) {
    pt d13=a0-b0;
    pt d43=b1-b0;
    pt d21=a1-a0;
    coord un = d43.x*d13.y - d43.y*d13.x;
    coord ud = d43.y*d21.x - d43.x*d21.y;
    if(abs(ud)<eps) return pt(oo,un);
    return pt(a0.x + un*d21.x/ud, a0.y + un*d21.y/ud);
}

/* Dijkstra. */
vector<int> dijkstra(int start, vector<vector<pair<int, int>>> &adj) {
    // adj contains pairs with {value, weight}
    priority_queue<pair<int, int>> pq;
    vector<int> dist(adj.size(), -1);
    pq.push({0, start});
    dist[start] = 0;

    while (!pq.empty()) {
        int u = pq.top().second; pq.pop();

        for (auto [v,w] : adj[u]) {
            if (dist[v] > -1 && dist[v] <= dist[u] + w) continue;
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
    }
    return dist;
}
