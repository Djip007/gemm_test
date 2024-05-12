#include <cstdio>
#include <iostream>

#include <ctime>
#include <cerrno>
#include <unistd.h>

inline int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

inline float float01(unsigned x) { // (0,1)
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

inline float numba(void) { // (-1,1)
    return float01(rand32()) * 2 - 1;
}

template <typename T> void clean(int m, int n, T *X) {
    for (int ij = 0; ij < m*n; ++ij)
        X[ij] = 1;
}

template <typename T> void control(int m, int n, T *X, T value) {
    for (int ij = 0; ij < m*n; ++ij) {
        if (X[ij] != value) {
            std::cout << "Erreur C["<< ij/m << "," << ij%m << "] = " << X[ij] << std::endl;
        }
    }
}

template <typename T> void randomize(int m, int n, T *X) {
    for (int ij = 0; ij < m*n; ++ij)
        X[ij] = numba();
}

template <typename T> T *l_malloc(int m, int n) {
    void *ptr;
    size_t size = sizeof(T) * m * n;
    if ((errno = posix_memalign(&ptr, sysconf(_SC_PAGESIZE), size))) {
        perror("posix_memalign");
        exit(1);
    }
    return (T *)ptr;
}

template <typename T> T *new_test_matrix(int m, int n) {
    T *X = l_malloc<T>(m, n);
    randomize(m, n, X);
    clean(m, n, X);
    return X;
}

struct mesure {
    struct timespec ts_0;
    void start() {
        clock_gettime(CLOCK_REALTIME, &ts_0);
    }
    double end() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        auto _s  = ts.tv_sec - ts_0.tv_sec;
        auto _ns = ts.tv_nsec - ts_0.tv_nsec;
        return ((double) _s) + ((double) _ns)/1.e9;
    }
};

#define BENCH(x) \
    do { \
        x; \
        mesure time; time.start(); \
        for (long long i = 0; i < ITERATIONS; ++i) { \
            asm volatile("" ::: "memory"); \
            x; \
            asm volatile("" ::: "memory"); \
        } \
        auto dt = time.end(); \
        printf("%g us %s %g gigaflops\n", (dt/ITERATIONS)*1000000, #x, (1e-9*2*m*n*k*ITERATIONS)/dt); \
    } while (0)


