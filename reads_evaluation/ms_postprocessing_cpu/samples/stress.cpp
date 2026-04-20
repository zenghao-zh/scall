/*******************************************************************************
 * FILENAME:      stress.cpp  (CPU port)
 *
 * Performance stress test — uses random qscore data.
 * cudaEvent timing replaced with std::chrono.
 *******************************************************************************/

#include "npy.h"
#include "rta_postprocess.h"
#include "spdlog/spdlog.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
  void* handle;
  auto  gpu_id = strtol(argv[1], nullptr, 10);

  RTAPostprocessParams postprocess_params;
  postprocess_params.batch_size = 256;
  CCA::Init(&handle, postprocess_params, static_cast<int>(gpu_id));

  int                buffer_chunk_header_offset = 15;
  std::vector<float> buffer_chunk_header_tmp(postprocess_params.batch_size *
                                             buffer_chunk_header_offset);
  CycloneAcc::loadNpy(argv[2],
                      reinterpret_cast<void**>(&buffer_chunk_header_tmp),
                      buffer_chunk_header_tmp.size(), 0);

  // Random qscore generator (same seed as original)
  std::mt19937                    mersenne_engine{1};
  std::uniform_int_distribution<int> dist{0, 73};
  auto gen = [&]() { return static_cast<char>(dist(mersenne_engine)); };

  std::vector<BufferChunk> buffer_chunk(postprocess_params.batch_size);
  for (int i = 0; i < postprocess_params.batch_size; i++)
  {
    buffer_chunk[i].chip_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 0]);
    buffer_chunk[i].chn_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 1]);
    buffer_chunk[i].chn_status =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 2]);
    buffer_chunk[i].cycle_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 3]);
    buffer_chunk[i].read_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 4]);
    buffer_chunk[i].chunk_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 5]);
    buffer_chunk[i].overlap =
      static_cast<int64_t>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 6]);
    buffer_chunk[i].offset =
      static_cast<int64_t>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 7]);
    buffer_chunk[i].start_idx =
      static_cast<int64_t>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 8]);
    buffer_chunk[i].openpore_before_median =
      buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 9];
    buffer_chunk[i].openpore_before_std =
      buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 10];
    buffer_chunk[i].sensor_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 11]);
    buffer_chunk[i].chunk_err =
      buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 12];
    buffer_chunk[i].chunk_base_num =
      static_cast<int>(buffer_chunk_header_tmp[i * buffer_chunk_header_offset + 13]);
  }

  int max_chunk_base_num = strtol(argv[3], nullptr, 10);
  int trim_head_base_num = strtol(argv[4], nullptr, 10);
  int trim_tail_base_num = strtol(argv[5], nullptr, 10);
  int iterations         = strtol(argv[6], nullptr, 10);

  for (int i = 0; i < iterations; i++)
  {
    std::vector<char> qscode_vec(
      postprocess_params.batch_size * postprocess_params.qstring_len, 0);
    std::generate(qscode_vec.begin(), qscode_vec.end(), gen);

    auto t0 = std::chrono::high_resolution_clock::now();

    CCA::PostProcess(handle, buffer_chunk.data(), qscode_vec.data(),
                     max_chunk_base_num, trim_head_base_num,
                     trim_tail_base_num);

    auto  t1  = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration<float, std::milli>(t1 - t0).count();
    SPDLOG_LOGGER_INFO(spdlog::default_logger(), "cost time: {} ms", ms);
  }

  return 0;
}
