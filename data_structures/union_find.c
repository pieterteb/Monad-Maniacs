#include <stdlib.h>

typedef struct UF {
    int* p;
    size_t *r, s;
} UF;

UF* uf_new(int N) {
    UF* uf = malloc(sizeof(*uf));
    uf->p = malloc(N * sizeof(*uf->p));
    uf->r = calloc(N, sizeof(*uf->r));
    for (int i = 0; i < N; ++i) uf->p[i] = i;
    uf->s = N;
    return uf;
}

int uf_root(UF* uf, int u) {
    int r = uf->p[u];
    if (uf->p[r] != r) return uf->p[u] = uf_find(uf, r);
    return r;
}

void uf_union(UF* uf, int u, int v) {
    int ru = uf_find(uf, u);
    int rv = uf_find(uf, v);
    if (ru == rv) return;
    --uf->s;
    if (uf->r[ru] < uf->r[rv]) {
        uf->p[ru] = rv;
    } else if (uf->r[ru] > uf->r[rv]) {
        uf->p[rv] = ru;
    } else {
        uf->p[rv] = ru;
        ++uf->r[ru];
    }
}

int uf_sets(UF* uf) {
    return uf->s;
}
