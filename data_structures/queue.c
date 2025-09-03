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
