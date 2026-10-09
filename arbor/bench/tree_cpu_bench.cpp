#include <cstdio>
#include <cstdlib>

#include "q8.h"

struct Leaf { Mat gate_up, down; };

static Leaf make_leaf(int d, int leaf, std::mt19937 &g) { return {make_mat(2 * leaf, d, g), make_mat(d, leaf, g)}; }

struct Layer { Mat qkv, o, router; std::vector<Leaf> leaves; Leaf shared; bool has_shared; };

static void token(const std::vector<Layer> &L, const Mat &head, const int *sel, int nsel,
                  const BlockQ8 *xq, float *y) {
#pragma omp parallel
    for (size_t l = 0; l < L.size(); l++) {
        const Layer &ly = L[l];
        matvec(ly.qkv, xq, y);
        matvec(ly.o, xq, y);
        matvec(ly.router, xq, y);
        if (ly.has_shared) {
            matvec(ly.shared.gate_up, xq, y);
            matvec(ly.shared.down, xq, y);
        }
        for (int s = 0; s < nsel; s++) {
            const Leaf &lf = ly.leaves[sel[l * nsel + s]];
            matvec(lf.gate_up, xq, y);
            matvec(lf.down, xq, y);
        }
    }
#pragma omp parallel
    matvec(head, xq, y);
}

static double run(bool shared, int d, int W, int layers, int ntok, std::mt19937 &g) {

    const int leaf = (W / 4 + QK - 1) / QK * QK, nleaf = 16, nsel = shared ? 3 : 4;
    std::vector<Layer> L;
    for (int l = 0; l < layers; l++) {
        Layer ly{make_mat(3 * d, d, g), make_mat(d, d, g), make_mat(32, d, g), {}, {}, shared};
        for (int e = 0; e < nleaf; e++) ly.leaves.push_back(make_leaf(d, leaf, g));
        if (shared) ly.shared = make_leaf(d, leaf, g);
        L.push_back(std::move(ly));
    }
    Mat head = make_mat(1280, d, g);
    std::vector<float> xf(2 * W, 0.1f), y(3 * d + 2 * W);
    std::vector<BlockQ8> xq(2 * W / QK);
    quantize_row(xf.data(), xq.data(), 2 * W);

    std::vector<int> sel((size_t)ntok * layers * nsel);
    for (int t = 0; t < ntok; t++)
        for (int l = 0; l < layers; l++) {
            std::vector<int> p(nleaf);
            for (int i = 0; i < nleaf; i++) p[i] = i;
            std::shuffle(p.begin(), p.end(), g);
            for (int s = 0; s < nsel; s++) sel[((size_t)t * layers + l) * nsel + s] = p[s];
        }
    const int warm = 50;
    double t0 = 0;
    for (int t = 0; t < ntok; t++) {
        if (t == warm) t0 = now();
        token(L, head, &sel[(size_t)t * layers * nsel], nsel, xq.data(), y.data());
    }
    return (ntok - warm) / (now() - t0);
}

int main(int argc, char **argv) {
    int ntok = argc > 1 ? atoi(argv[1]) : 600;
    printf("threads %d, tokens %d\n", omp_get_max_threads(), ntok);
    struct Cfg { const char *name; int d, W, layers; } cfgs[] = {{"S", 512, 1344, 8}, {"M", 768, 2048, 12}};
    for (auto &c : cfgs) {
        for (int rep = 0; rep < 3; rep++) {
            std::mt19937 g(rep);
            double a3 = run(false, c.d, c.W, c.layers, ntok, g);
            double a3s = run(true, c.d, c.W, c.layers, ntok, g);
            printf("%s rep%d  A3 %.0f tok/s  A3s %.0f tok/s  A3s/A3 %.3f\n", c.name, rep, a3, a3s, a3s / a3);
        }
    }
    return 0;
}
