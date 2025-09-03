#include <stdlib.h>

#define q(T) struct { \
    T* a; \
    size_t f, b, cap; \
}

typedef q(int) qi;

#define q_new(T) (malloc(sizeof(q(T))))
#define qi_new() (q_new(int))

#define q_init(q, N) do { \
    (q)->a = malloc((N) * sizeof(*(q)->a)); \
    (q)->f = 0; \
    (q)->b = 0; \
    (q)->cap = (N); \
} while (0)

#define q_push(q, x) do { \
    (q)->a[(q)->b] = (x); \
    (q)->b = (++(q)->b) % (q)->cap; \
} while (0)

#define q_pop(q) ((q)->a[(q)->f++])

#define q_front(q) ((q)->a[(q)->f])

#define q_empty(q) ((q)->f == (q)->b)

#define q_destroy(q) do { \
    free((q)->a); \
    free(q); \
} while (0)

void bfs(int** adj, size_t* adjc, int* seen, int N) {
    qi* q = qi_new();
    q_init(q, N);

    int u, v, j;
    for (int i = 0; i < N; ++i) {
        if (seen[i]) continue;

        q_push(q, i);
        while (!q_empty(q)) {
            u = q_pop(q);
            for (j = 0; j < adjc[u]; ++j) {
                v = adj[u][j];
                if (!seen[v]) {
                    q_push(q, v);
                    seen[v] = 1;
                }
            }
        }
    }

    q_destroy(q);
}
