#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <vector>
#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#include "cnpy.h"

const int N_BINS = 64;
const float BIN_OFFSET = 0 * M_PI / 180.0;
const float BIN_WIDTH = 0.5 * M_PI / 180.0;

int gpu0;

#define CUDA_CHECK(expr) do {                                                                      \
  cudaError_t rc = (expr);                                                                         \
  if (rc != cudaSuccess) {                                                                      \
        printf("CUDA error %d '%s' - %s:%d: %s", rc, cudaGetErrorString(rc),  __FILE__, __LINE__, #expr); \
        exit(-1); \
  } \
} while(0)

struct point {
    vec3f pos;
    float value;
};

struct query {
    int32_t idx;
    vec3f pos;
    float cnt[N_BINS], sum[N_BINS], sum2[N_BINS];
};

__global__ void compute_histograms_kernel(int n_data, point* data, int n_queries, query* queries) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float cnt[N_BINS] = {};
    float sum[N_BINS] = {};
    float sum2[N_BINS] = {};

    if (query_idx < n_queries) {
        vec3f q = queries[query_idx].pos;
        int from = (n_data / blockDim.y) * threadIdx.y;
        int to = min(from + n_data / blockDim.y, n_data);
        for (int i = from; i < to; ++i) {
            vec3f d = data[i].pos;
            float v = data[i].value;
            float angle = (acosf(q.x * d.x + q.y * d.y + q.z * d.z) - BIN_OFFSET) / BIN_WIDTH;
            if (0 <= angle && angle < N_BINS) {
                int at = (int)angle;
                cnt[at] += 1.0f;
                sum[at] += v;
                sum2[at] += v * v;
            }
        }
        for (int i = 0; i < N_BINS; ++i) {
            atomicAdd(&queries[query_idx].cnt[i], cnt[i]);
            atomicAdd(&queries[query_idx].sum[i], sum[i]);
            atomicAdd(&queries[query_idx].sum2[i], sum2[i]);
        }
    }
}

void compute_histograms(int n_data, point* data, int n_queries, query* queries) {
    // CUDA_CHECK(cudaMemPrefetchAsync(data, n_data * sizeof(point), gpu0));
    // CUDA_CHECK(cudaMemPrefetchAsync(queries, n_queries * sizeof(query), gpu0));
    compute_histograms_kernel<<<(n_queries + 31) / 32, dim3(32, 32)>>>(n_data, data, n_queries, queries);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    CUDA_CHECK(cudaGetDevice(&gpu0));

    auto base = read_Healpix_map_from_fits<float>("wmap_band_imap_r9_9yr_W_v5.fits");
    // auto base = read_Healpix_map_from_fits<float>("COM_CMB_IQU-smica_2048_R3.00_full.fits");
    // auto base = read_Healpix_map_from_fits<float>("commander_smooth.fits");
    // auto base = read_Healpix_map_from_fits<float>("sim/sim0.fits");
    point* data;
    CUDA_CHECK(cudaMallocManaged(&data, base.Npix() * sizeof(point)));
    int n_data = 0;
    float cutoff = asinf(20.0 * M_PI / 180.0);
    for (int i = 0; i < base.Npix(); ++i) {
        auto p = (vec3f)base.pix2vec(i);
        if (p.z >= cutoff) {
            data[n_data].value = base[i] * 1e6;
            data[n_data++].pos = p;
        }
    }
    int n_data_upper = n_data;
    for (int i = 0; i < base.Npix(); ++i) {
        auto p = (vec3f)base.pix2vec(i);
        if (p.z <= -cutoff) {
            data[n_data].value = base[i] * 1e6;
            data[n_data++].pos = p;
        }
    }

    Healpix_Base queries_base(8, Healpix_Ordering_Scheme::NEST);
    query* queries;
    CUDA_CHECK(cudaMallocManaged(&queries, queries_base.Npix() * sizeof(query)));
    int n_queries = 0;
    for (int i = 0; i < queries_base.Npix(); ++i) {
        auto p = (vec3f)queries_base.pix2vec(i);
        if (p.z >= cutoff) {
            queries[n_queries] = {};
            queries[n_queries].idx = i;
            queries[n_queries++].pos = p;
        }
    }
    int n_queries_upper = n_queries;
    for (int i = 0; i < queries_base.Npix(); ++i) {
        auto p = (vec3f)queries_base.pix2vec(i);
        if (p.z <= -cutoff) {
            queries[n_queries] = {};
            queries[n_queries].idx = i;
            queries[n_queries++].pos = p;
        }
    }

    compute_histograms(n_data_upper, data, n_queries_upper, queries);
    compute_histograms(n_data - n_data_upper, data + n_data_upper, n_queries - n_queries_upper, queries + n_queries_upper);

    // double baselines[N_BINS] = {};
    // for (int i = 0; i < n_queries; ++i) {
    //     for (int bin = 0; bin < N_BINS; ++bin) {
    //         double n = queries[i].cnt[bin];
    //         double mean = queries[i].sum[bin] / n;
    //         double sigma = sqrt((queries[i].sum2[bin] - n * mean * mean) / (n - 1));
    //         baselines[bin] += sigma / n_queries;
    //     }
    // }

    Healpix_Map<float> baseline_map(8, Healpix_Ordering_Scheme::NEST);
    baseline_map.fill(Healpix_undef);

    std::vector<double> all_baselines;
    std::vector<double> circle_means[N_BINS], circle_sigmas[N_BINS];
    std::vector<double> circle_cnts, circle_sums, circle_sums2;
    int counts[N_BINS] = {};
    int hist2[N_BINS + 1] = {};
    for (int i = 0; i < n_queries; ++i) {
        double sigmas[N_BINS], means[N_BINS], mean_sigma = 0.0;
        double disk_sum = 0.0, disk_sum2 = 0.0, disk_cnt = 0.0;
        for (int bin = 0; bin < N_BINS; ++bin) {
            circle_cnts.push_back(queries[i].cnt[bin]);
            circle_sums.push_back(queries[i].sum[bin]);
            circle_sums2.push_back(queries[i].sum2[bin]);
            double n = queries[i].cnt[bin];
            double mean = queries[i].sum[bin] / n;
            double sigma = sqrt((queries[i].sum2[bin] - n * mean * mean) / (n - 1));
            means[bin] = mean;
            sigmas[bin] = sigma;
            circle_means[bin].push_back(mean);
            circle_sigmas[bin].push_back(sigma);
            if (32 <= bin && bin <= 64) {
                mean_sigma += sigma;
            }
            if (5 <= bin && bin <= 30) {
                disk_cnt += queries[i].cnt[bin];
                disk_sum += queries[i].sum[bin];
                disk_sum2 += queries[i].sum2[bin];
            }
        }
        mean_sigma /= 32;

        // double disk_mean = disk_sum / disk_cnt;
        // mean_sigma = sqrt((disk_sum2 - disk_cnt * disk_mean * disk_mean) / (disk_cnt - 1));

        all_baselines.push_back(mean_sigma);
        baseline_map[queries[i].idx] = mean_sigma;

        // double sigmas2[N_BINS];
        // std::memcpy(sigmas2, sigmas, sizeof(sigmas));
        // std::sort(sigmas2, sigmas2 + N_BINS);
        // mean_sigma = sigmas2[N_BINS / 2];
        double baselines[N_BINS] = {};
        for (int bin = 0; bin < N_BINS; ++bin) {
            // double sum = 0.0, cnt = 0.0;
            // for (int db = max(0, bin - 8); db <= bin + 8 && db < N_BINS; ++db) {
            //     sum += sigmas[db];
            //     cnt++;
            // }
            // baselines[bin] = sum / cnt;
            baselines[bin] = mean_sigma;
        }

        int circles_found = 0;
        for (int bin = 0; bin < N_BINS; ++bin) {
            if (sigmas[bin] < baselines[bin] - 0.020) {
                circles_found += 1;
                while (bin + 1 < N_BINS && sigmas[bin + 1] < baselines[bin + 1]) {
                    bin++;
                }
            }
        }

        for (int bin = 0; bin < N_BINS; ++bin) {
            if (sigmas[bin] < baselines[bin] - 0.020) {
                counts[bin] += 1;
            }
        }
        hist2[circles_found]++;
    }
    for (int bin = 0; bin < N_BINS; ++bin) {
        std::cerr << (double)counts[bin] / n_queries << " ";
    }
    std::cerr << "\n" << n_queries << "\n";
    // for (int bin = 0; bin < N_BINS; ++bin) {
    //     std::cerr << baselines[bin] << " ";
    // }
    // std::cerr << "\n";
    for (int bin = 0; bin <= 10; ++bin) {
        std::cerr << hist2[bin] << " ";
    }

    cnpy::npz_save("sim/out.npz", "counts", counts, {N_BINS});
    cnpy::npz_save("sim/out.npz", "hist2", hist2, {N_BINS + 1}, "a");
    cnpy::npz_save("sim/out.npz", "baselines", all_baselines, "a");
    cnpy::npz_save("sim/out.npz", "cnts", circle_cnts.data(), {(size_t)n_queries, N_BINS}, "a");
    cnpy::npz_save("sim/out.npz", "sums", circle_sums.data(), {(size_t)n_queries, N_BINS}, "a");
    cnpy::npz_save("sim/out.npz", "sums2", circle_sums2.data(), {(size_t)n_queries, N_BINS}, "a");
    for (int i = 0; i < N_BINS; ++i) {
        cnpy::npz_save("sim/out.npz", "circle" + std::to_string(i) + "_mean", circle_means[i], "a");
        cnpy::npz_save("sim/out.npz", "circle" + std::to_string(i) + "_sigma", circle_sigmas[i], "a");
    }

    write_Healpix_map_to_fits("baseline.fits", baseline_map, PDT::PLANCK_FLOAT32);

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(queries));
    
    return 0;
}
