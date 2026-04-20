/*******************************************************************************
 * FILENAME:      parse_dat.cpp  (CPU port)
 *
 * Reads binary .dat BufferChunk file + qscore .npy, runs PostProcess,
 * writes results to CSV.
 *
 * CLI:  parse_dat <gpu_id> <chunk.dat> <qscore.npy>
 *                <max_chunk_base_num> <trim_head> <trim_tail>
 *                <output.csv> <iterations>
 *******************************************************************************/

#include "npy.h"
#include "rta_postprocess.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
  void* handle;
  auto  gpu_id = strtol(argv[1], nullptr, 10);  // accepted, ignored on CPU

  RTAPostprocessParams postprocess_params;
  postprocess_params.batch_size = 256;
  CCA::Init(&handle, postprocess_params, static_cast<int>(gpu_id));

  // ── Load binary .dat file (array of BufferChunk structs) ──────────────────
  std::ifstream fin(argv[2], std::ios::binary);
  if (!fin) { std::cerr << "Failed to open dat file\n"; return 1; }

  fin.seekg(0, std::ios::end);
  size_t filesize = static_cast<size_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);

  size_t chunkSize = sizeof(BufferChunk);
  size_t numChunks = filesize / chunkSize;

  std::vector<BufferChunk> chunks(numChunks);
  for (size_t i = 0; i < numChunks; i++)
  {
    fin.read(reinterpret_cast<char*>(&chunks[i]), static_cast<std::streamsize>(chunkSize));
    if (!fin) { std::cerr << "Error reading chunk " << i << "\n"; break; }
  }
  fin.close();

  // ── Load qscore .npy ───────────────────────────────────────────────────────
  std::vector<char> qscode_vec(256 * postprocess_params.qstring_len, 0);
  auto qscode_vec_ptr = qscode_vec.data();
  CycloneAcc::loadNpy<char>(argv[3],
                            reinterpret_cast<void**>(&qscode_vec_ptr),
                            qscode_vec.size(), 0);

  int max_chunk_base_num = strtol(argv[4], nullptr, 10);
  int trim_head_base_num = strtol(argv[5], nullptr, 10);
  int trim_tail_base_num = strtol(argv[6], nullptr, 10);

  // ── Create output directory ───────────────────────────────────────────────
  fs::create_directories(fs::path(argv[7]).parent_path());

  int iterations = strtol(argv[8], nullptr, 10);

  for (int i = 0; i < iterations; i++)
  {
    auto t0 = std::chrono::high_resolution_clock::now();

    CCA::PostProcess(handle, chunks.data(), qscode_vec_ptr,
                     max_chunk_base_num, trim_head_base_num,
                     trim_tail_base_num);

    auto  t1  = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cout << "[timer] iteration " << i << " cost: " << ms << " ms\n";
  }

  // ── Write results CSV ──────────────────────────────────────────────────────
  std::ofstream file(argv[7]);
  for (const auto& chunk : chunks)
  {
    file << chunk.chip_num   << "," << chunk.chn_num    << ","
         << chunk.chn_status << "," << chunk.cycle_num  << ","
         << chunk.read_num   << "," << chunk.chunk_num  << ","
         << chunk.overlap    << "," << chunk.offset     << ","
         << chunk.start_idx  << "," << chunk.openpore_before_median << ","
         << chunk.openpore_before_std  << ","
         << chunk.read_median          << ","
         << chunk.read_std             << ","
         << chunk.sensor_num           << ","
         << chunk.chunk_err            << ","
         << chunk.chunk_base_num       << ","
         << chunk.chunk_type           << ","
         << chunk.trim_err_type        << "\n";
  }

  return 0;
}
