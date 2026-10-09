#pragma once
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr int QK = 32;
struct BlockQ8 { float d; int8_t q[QK]; };

static void quantize_row(const float *x, BlockQ8 *y, int k) {
    for (int b = 0; b < k / QK; b++) {
        float amax = 0;
        for (int j = 0; j < QK; j++) amax = std::max(amax, std::fabs(x[b * QK + j]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        y[b].d = d;
        for (int j = 0; j < QK; j++) y[b].q[j] = (int8_t)std::lrintf(x[b * QK + j] * id);
    }
}

static inline float dot_q8(const BlockQ8 *w, const BlockQ8 *x, int nb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        __m256i qw = _mm256_loadu_si256((const __m256i *)w[b].q);
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[b].q);

        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qx, qw));
        __m256 s = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(w[b].d * x[b].d), s, acc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_hadd_ps(h, h); h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}

struct Mat { int n, k; std::vector<BlockQ8> w; };

static Mat make_mat(int n, int k, std::mt19937 &g) {
    Mat m{n, k, std::vector<BlockQ8>((size_t)n * k / QK)};
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> row(k);
    for (int i = 0; i < n; i++) {
        for (auto &v : row) v = nd(g);
        quantize_row(row.data(), &m.w[(size_t)i * k / QK], k);
    }
    return m;
}

static void matvec(const Mat &m, const BlockQ8 *x, float *y) {
    const int nb = m.k / QK;
#pragma omp for schedule(static)
    for (int i = 0; i < m.n; i++) y[i] = dot_q8(&m.w[(size_t)i * nb], x, nb);
}

static double bytes(const Mat &m) { return (double)m.w.size() * sizeof(BlockQ8); }
