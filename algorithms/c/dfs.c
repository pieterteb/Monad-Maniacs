#include <stdlib.h>

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

void dfs(int** adj, size_t* adjc, int* seen, size_t N) {
    sti* st = sti_new();
    st_init(st, N);

    int u, v, i;
    for (int i = 0; i < N; ++i) {
        if (seen[i]) continue;

        st_push(st, i);
        while (!st_empty(st)) {
            u = st_pop(st);
            for (i = 0; i < adjc[u]; ++i) {
                v = adj[u][i];
                if (!seen[v]) {
                    st_push(st, v);
                    seen[v] = 1;
                }
            }
        }
    }

    st_destroy(st);
}
