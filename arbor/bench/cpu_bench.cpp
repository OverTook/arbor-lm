#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#include "q8.h"

static void bench_triad() {
    const size_t n = 1ull << 27;
    double *a = (double *)aligned_alloc(64, n * 8), *b = (double *)aligned_alloc(64, n * 8),
           *c = (double *)aligned_alloc(64, n * 8);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }
    double best_copy = 1e9, best_triad = 1e9;
    for (int r = 0; r < 10; r++) {
        double t = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i];
        best_copy = std::min(best_copy, now() - t);
        t = now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        best_triad = std::min(best_triad, now() - t);
    }

    printf("stream_copy_GBps %.1f\n", 2.0 * n * 8 / best_copy / 1e9);
    printf("stream_triad_GBps %.1f\n", 3.0 * n * 8 / best_triad / 1e9);
    if (a[n / 2] != 7.0) printf("triad check FAILED\n");
    free(a); free(b); free(c);
}

static void check_dot(std::mt19937 &g) {
    Mat m = make_mat(4, 768, g);
    std::vector<float> xf(768); std::normal_distribution<float> nd(0, 1);
    for (auto &v : xf) v = nd(g);
    std::vector<BlockQ8> xq(768 / QK); quantize_row(xf.data(), xq.data(), 768);
    for (int i = 0; i < 4; i++) {
        double ref = 0;
        for (int b = 0; b < 768 / QK; b++) {
            int s = 0;
            for (int j = 0; j < QK; j++) s += m.w[i * 24 + b].q[j] * xq[b].q[j];
            ref += (double)m.w[i * 24 + b].d * xq[b].d * s;
        }
        float got = dot_q8(&m.w[i * 24], xq.data(), 24);
        if (std::fabs(got - ref) > 1e-4 * (1 + std::fabs(ref))) { printf("dot check FAILED %f %f\n", got, ref); exit(1); }
    }
    printf("dot_check ok\n");
}

int main() {
    printf("threads %d\n", omp_get_max_threads());
    std::mt19937 g(0);
    check_dot(g);
    bench_triad();

    {
        Mat big = make_mat(138240, 768, g);
        std::vector<float> xf(768, 0.1f), y(big.n);
        std::vector<BlockQ8> xq(768 / QK); quantize_row(xf.data(), xq.data(), 768);
        double best = 1e9;
        for (int r = 0; r < 20; r++) {
            double t = now();
#pragma omp parallel
            matvec(big, xq.data(), y.data());
            best = std::min(best, now() - t);
        }
        printf("q8_big_MB %.1f  GBps %.1f  tok_per_s_equiv %.0f\n", bytes(big) / 1e6, bytes(big) / best / 1e9, 1 / best);
    }

    {
        const int d = 768, f = 2048, L = 12, V = 32000;
        std::vector<Mat> mats;
        for (int l = 0; l < L; l++) {
            for (int j = 0; j < 4; j++) mats.push_back(make_mat(d, d, g));
            mats.push_back(make_mat(f, d, g)); mats.push_back(make_mat(f, d, g));
            mats.push_back(make_mat(d, f, g));
        }
        mats.push_back(make_mat(V, d, g));
        double total = 0; for (auto &m : mats) total += bytes(m);
        std::vector<float> y(V), xf(f, 0.1f);
        std::vector<BlockQ8> xq(f / QK); quantize_row(xf.data(), xq.data(), f);
        double best = 1e9;
        for (int r = 0; r < 20; r++) {
            double t = now();
#pragma omp parallel
            for (auto &m : mats) matvec(m, xq.data(), y.data());
            best = std::min(best, now() - t);
        }
        printf("q8_layered_MB %.1f  GBps %.1f  tok_per_s_equiv %.0f  matvecs %zu\n",
               total / 1e6, total / best / 1e9, 1 / best, mats.size());
    }
    return 0;
}
