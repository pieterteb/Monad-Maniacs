#include <stdlib.h>

#define MIN(x, y)   ((x) < (y) ? (x) : (y))

#define pair(T, Y) struct { \
    T f; \
    Y s; \
}

typedef pair(int, int) pairii;

#define st(T) struct { \
    T* a; \
    size_t i, cap; \
}

typedef st(int) sti;

#define st_new(T) (malloc(sizeof(st(T))))
#define sti_new() (st_new(int))

#define st_init(st, N) do { \
    (st)->a = malloc((N) * sizeof(*(st)->a)); \
    (st)->i = 0; \
    (st)->cap = (N); \
} while (0)

#define st_push(st, x) do { \
    if ((st)->i == (st)->cap) { \
        (st)->cap *= 2; \
        (st)->a = realloc((st)->a, (st)->cap * sizeof(*(st)->a)); \
    } \
    (st)->a[(st)->i++] = (x); \
} while (0)

#define st_pop(st) ((st)->a[--(st)->i])

#define st_top(st) ((st)->a[(st)->i - 1])

#define st_empty(st) ((st)->i == 0)

#define st_destroy(st) do { \
    free((st)->a); \
    free(st); \
} while (0)

int scc(int** adj, size_t* adjc, int** comp, int* compc, size_t N) {
    st(pairii)* dfs = st_new(pairii); st_init(dfs, N);
    sti* st = sti_new(); st_init(st, N);
    int* num = malloc(N * sizeof(*num));
    int* low = malloc(N * sizeof(*low));
    *comp = realloc(*comp, N * sizeof(**comp));
    for (size_t i = 0; i < N; ++i) num[i] = (*comp)[i] = -1;
    int curnum = 0; *compc = 0;

    int u, v, pi, cur; pairii p;
    int root;
    for (int i = 0; (unsigned int)i < N; ++i) {
        if (num[i] != -1) continue;
        st_push(dfs, ((pairii){ i, 0 })); low[i] = curnum++;
        while (!st_empty(dfs)) {
            p = st_pop(dfs);
            u = p.f; pi = p.s;
            if (!pi) {
                num[u] = low[u] = curnum++;
                st_push(st, u);
            } else if (pi > 0) {
                v = adj[u][pi - 1];
                low[u] = MIN(low[u], low[v]);
            }
            while (pi < adjc[u] && num[adj[u][pi]] != -1) {
                v = adj[u][pi++];
                if ((*comp)[v] == -1)
                    low[u] = MIN(low[u], num[v]);
            }
            if (pi < adjc[u]) {
                v = adj[u][pi];
                st_push(dfs, ((pairii){ u, pi + 1 }));
                st_push(dfs, ((pairii){ v, 0 }));
            } else if (low[u] == num[u]) {
                do {
                    cur = st_pop(st);
                    (*comp)[cur] = *compc;
                } while (cur != u);
                ++*compc;
            }
        }
    }
    st_destroy(dfs); st_destroy(st); free(num); free(low);
}
