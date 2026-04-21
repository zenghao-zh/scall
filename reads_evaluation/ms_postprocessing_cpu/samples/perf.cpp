/*******************************************************************************
 * perf.cpp — ms_postprocessing CPU throughput benchmark
 *
 * Measures throughput as "Q-score characters processed per second",
 * reported in M/s (= 1e6 chars/s). The workload per iteration is
 *   batch_size × qstring_len  characters.
 *
 * Usage:
 *   ./Release/perf <batch_size> <qstring_len> <warmup> <iterations>
 *
 * Example:
 *   ./Release/perf 256 1000 5 200
 *******************************************************************************/

#include "rta_postprocess.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
  int batch_size  = (argc > 1) ? std::atoi(argv[1]) : 256;
  int qstring_len = (argc > 2) ? std::atoi(argv[2]) : 1000;
  int warmup      = (argc > 3) ? std::atoi(argv[3]) : 5;
  int iterations  = (argc > 4) ? std::atoi(argv[4]) : 200;

  // Init handle
  void* handle;
  RTAPostprocessParams params;
  params.batch_size  = batch_size;
  params.qstring_len = qstring_len;
  params.chunk_size  = 5000;
  CCA::Init(&handle, params, 0);

  // Build random but deterministic Q-score buffer (Phred33: 33..73 range is typical)
  std::vector<char> qscode(static_cast<size_t>(batch_size) * qstring_len);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(33, 73);
  for (auto& c : qscode) c = static_cast<char>(dist(rng));

  // Build BufferChunk headers — use a mix of short reads (chunk_num=0)
  // and long reads (chunk_num > 0) so both code paths get exercised.
  std::vector<BufferChunk> buffer_chunk(batch_size);
  for (int i = 0; i < batch_size; i++)
  {
    buffer_chunk[i].chip_num    = 0;
    buffer_chunk[i].chn_num     = i;
    buffer_chunk[i].chn_status  = 1;
    buffer_chunk[i].cycle_num   = 1;
    buffer_chunk[i].read_num    = 1;
    buffer_chunk[i].chunk_num   = (i % 4 == 0) ? 0 : (1 + (i % 5));
    buffer_chunk[i].overlap     = (buffer_chunk[i].chunk_num == 0) ? 5000 : 500;
    buffer_chunk[i].offset      = 0;
    buffer_chunk[i].start_idx   = 0;
    buffer_chunk[i].sensor_num  = i % 4;
    buffer_chunk[i].chunk_err   = 0.f;
    buffer_chunk[i].chunk_base_num = 0;
    buffer_chunk[i].trim_err_type  = 0;
  }

  const int  stride             = 5;
  const int  max_chunk_base_num = 100;
  const int  trim_head_base_num = 40;
  const int  trim_tail_base_num = 40;

  // Warmup
  for (int i = 0; i < warmup; i++)
  {
    CCA::PostProcess(handle, buffer_chunk.data(), qscode.data(),
                     max_chunk_base_num, trim_head_base_num,
                     trim_tail_base_num, stride);
  }

  // Benchmark
  std::vector<double> times_ms;
  times_ms.reserve(iterations);

  auto total_t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    CCA::PostProcess(handle, buffer_chunk.data(), qscode.data(),
                     max_chunk_base_num, trim_head_base_num,
                     trim_tail_base_num, stride);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(ms);
  }
  auto total_t1 = std::chrono::high_resolution_clock::now();
  double total_s = std::chrono::duration<double>(total_t1 - total_t0).count();

  // Statistics
  std::sort(times_ms.begin(), times_ms.end());
  double sum = 0.0;
  for (auto v : times_ms) sum += v;
  double mean_ms   = sum / times_ms.size();
  double median_ms = times_ms[times_ms.size() / 2];
  double p99_ms    = times_ms[std::min(times_ms.size() - 1,
                                       static_cast<size_t>(times_ms.size() * 0.99))];
  double min_ms    = times_ms.front();
  double max_ms    = times_ms.back();

  // Throughput: (batch_size * qstring_len) chars per call
  const double chars_per_call = static_cast<double>(batch_size) * qstring_len;
  const double total_chars    = chars_per_call * iterations;
  const double mean_throughput = chars_per_call / (mean_ms * 1e-3);     // chars/s
  const double total_throughput = total_chars / total_s;                // chars/s

  std::printf("\n===== ms_postprocessing CPU performance =====\n");
  std::printf("  batch_size         : %d\n", batch_size);
  std::printf("  qstring_len        : %d\n", qstring_len);
  std::printf("  warmup / iterations: %d / %d\n", warmup, iterations);
  std::printf("  chars per call     : %.0f\n", chars_per_call);
  std::printf("  latency (ms)       : min=%.3f  mean=%.3f  median=%.3f  p99=%.3f  max=%.3f\n",
              min_ms, mean_ms, median_ms, p99_ms, max_ms);
  std::printf("  throughput (mean)  : %.3f M chars/s\n", mean_throughput / 1e6);
  std::printf("  throughput (total) : %.3f M chars/s\n", total_throughput / 1e6);
  std::printf("=============================================\n\n");

  return 0;
}
