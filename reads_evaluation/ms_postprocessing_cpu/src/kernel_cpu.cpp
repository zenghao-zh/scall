/*******************************************************************************
 * FILENAME:      kernel_cpu.cpp
 *
 * CPU port of kernel.cu.
 *
 * All __device__ helper functions become static inline functions.
 * The __global__ postprocess_kernel becomes a plain loop.
 * Batch-level parallelism is handled with OpenMP.
 *
 * Algorithm is bit-for-bit equivalent to the CUDA version when running on
 * the same data (both use IEEE-754 float, same operation order per element).
 *******************************************************************************/

#include "kernel_cpu.h"

#include <cmath>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions (replicate __device__ functions from kernel.cu)
// ─────────────────────────────────────────────────────────────────────────────

// Count non-zero bytes in qstring[0 .. qstring_offset)
static inline void
cal_base_num(const char* qstring, int64_t qstring_offset, int* nonzero)
{
  for (int64_t i = 0; i < qstring_offset; i++)
  {
    if (qstring[i] != 0)
      ++(*nonzero);
  }
}

// Accumulate Q error and count non-zero bases
static inline void
cal_q_error(const char* qstring,
            int64_t     qstring_offset,
            float*      err_val,
            int*        nonzero)
{
  for (int64_t i = 0; i < qstring_offset; i++)
  {
    int val = static_cast<int>(qstring[i]);
    if (val != 0)
    {
      *err_val += std::pow(10.0f, -static_cast<float>(val - 33) / 10.0f);
      ++(*nonzero);
    }
  }
}

// Accumulate Q error only (no count)
static inline void
cal_q_error(const char* qstring, int64_t qstring_offset, float* err_val)
{
  for (int64_t i = 0; i < qstring_offset; i++)
  {
    int val = static_cast<int>(qstring[i]);
    if (val != 0)
      *err_val += std::pow(10.0f, -static_cast<float>(val - 33) / 10.0f);
  }
}

// Accumulate Q error for bases in the index range (start_base_index, end_base_index]
static inline void
cal_q_error(const char* qstring,
            int64_t     qstring_offset,
            int         start_base_index,
            int         end_base_index,
            float*      err_val)
{
  int   nonzero = 0;
  float err     = 0.0f;
  for (int64_t i = 0; i < qstring_offset; i++)
  {
    int val = static_cast<int>(qstring[i]);
    if (val != 0)
    {
      err = std::pow(10.0f, -static_cast<float>(val - 33) / 10.0f);
      ++nonzero;
      if (nonzero > start_base_index && nonzero <= end_base_index)
        *err_val += err;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// cal_chunk_q — no base truncation
// ─────────────────────────────────────────────────────────────────────────────
static inline void
cal_chunk_q(int         chunk_idx,
            int64_t     chunk_overlap,
            const char* qstring,
            int         qstring_len,
            int         chunk_size,
            int         overlap,
            int         stride,
            float*      err_val,
            int*        nonzero)
{
  if (chunk_idx == 1)
  {
    auto semi_t_overlap  = overlap / (2 * stride);
    int  qstring_offset  = qstring_len - semi_t_overlap;
    cal_q_error(qstring, qstring_offset, err_val, nonzero);
    return;
  }

  int64_t     qstring_offset;
  const char* cur_qstring;
  if (chunk_overlap <= 0)
  {
    int stub        = chunk_size + static_cast<int>(chunk_overlap);
    qstring_offset  = (stub + overlap / 2) / stride;
    cur_qstring     = qstring + (qstring_len - qstring_offset);
  }
  else
  {
    auto semi_t_overlap = overlap / (2 * stride);
    qstring_offset      = qstring_len - 2 * semi_t_overlap;
    cur_qstring         = qstring + semi_t_overlap;
  }
  cal_q_error(cur_qstring, qstring_offset, err_val, nonzero);
}

// ─────────────────────────────────────────────────────────────────────────────
// cal_chunk_q — with base truncation
// ─────────────────────────────────────────────────────────────────────────────
static inline void
cal_chunk_q(int         chunk_idx,
            int64_t     chunk_overlap,
            const char* qstring,
            int         qstring_len,
            int         chunk_size,
            int         overlap,
            int         stride,
            int         max_chunk_base_num,
            int         trim_head_base_num,
            int         trim_tail_base_num,
            float*      err_val,
            int*        nonzero,
            int*        trim_err_type)
{
  if (chunk_idx == 1)
  {
    auto semi_t_overlap = overlap / (2 * stride);
    int  qstring_offset = qstring_len - semi_t_overlap;

    cal_base_num(qstring, qstring_offset, nonzero);
    if (*nonzero > max_chunk_base_num)
    {
      int start_base_index = trim_head_base_num;
      int end_base_index   = *nonzero;
      cal_q_error(qstring, qstring_offset, start_base_index, end_base_index,
                  err_val);
      *trim_err_type = 1;
    }
    else
    {
      cal_q_error(qstring, qstring_offset, err_val);
      *trim_err_type = 0;
    }
    return;
  }

  int64_t     qstring_offset;
  const char* cur_qstring;
  if (chunk_overlap <= 0)
  {
    int stub       = chunk_size + static_cast<int>(chunk_overlap);
    qstring_offset = (stub + overlap / 2) / stride;
    cur_qstring    = qstring + (qstring_len - qstring_offset);

    cal_base_num(cur_qstring, qstring_offset, nonzero);
    if (*nonzero > max_chunk_base_num)
    {
      int start_base_index = 0;
      int end_base_index   = *nonzero - trim_tail_base_num;
      cal_q_error(cur_qstring, qstring_offset, start_base_index, end_base_index,
                  err_val);
      *trim_err_type = 2;
    }
    else
    {
      cal_q_error(cur_qstring, qstring_offset, err_val);
      *trim_err_type = 0;
    }
  }
  else
  {
    auto semi_t_overlap = overlap / (2 * stride);
    qstring_offset      = qstring_len - 2 * semi_t_overlap;
    cur_qstring         = qstring + semi_t_overlap;
    cal_q_error(cur_qstring, qstring_offset, err_val, nonzero);
    *trim_err_type = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Process a single chunk element — mirrors the GPU kernel body
// ─────────────────────────────────────────────────────────────────────────────

// With base truncation
static inline void
process_one_with_trim(BufferChunkHeader* header,
                      const char*        qstr_it,
                      int                qstr_offset,
                      int                chunk_size,
                      int                max_chunk_base_num,
                      int                trim_head_base_num,
                      int                trim_tail_base_num,
                      int                stride)
{
  if (header->chunk_num == 0)
  {
    // Short reads
    auto cur_offset = header->overlap / stride;

    cal_base_num(qstr_it, cur_offset, &header->chunk_base_num);

    if (header->chunk_base_num > max_chunk_base_num)
    {
      int start_base_index = trim_head_base_num;
      int end_base_index   = header->chunk_base_num - trim_tail_base_num;
      cal_q_error(qstr_it, cur_offset, start_base_index, end_base_index,
                  &header->chunk_err);
      header->trim_err_type = 3;
    }
    else
    {
      cal_q_error(qstr_it, cur_offset, &header->chunk_err);
      header->trim_err_type = 0;
    }
  }
  else
  {
    // Long reads
    cal_chunk_q(header->chunk_num, header->overlap, qstr_it, qstr_offset,
                chunk_size, 500, stride, max_chunk_base_num,
                trim_head_base_num, trim_tail_base_num, &header->chunk_err,
                &header->chunk_base_num, &header->trim_err_type);
  }
}

// Without base truncation
static inline void
process_one_no_trim(BufferChunkHeader* header,
                    const char*        qstr_it,
                    int                qstr_offset,
                    int                chunk_size,
                    int                stride)
{
  if (header->chunk_num == 0)
  {
    auto cur_offset = header->overlap / stride;
    cal_q_error(qstr_it, cur_offset, &header->chunk_err,
                &header->chunk_base_num);
  }
  else
  {
    cal_chunk_q(header->chunk_num, header->overlap, qstr_it, qstr_offset,
                chunk_size, 500, stride, &header->chunk_err,
                &header->chunk_base_num);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

int native_postprocess(BufferChunkHeader* buffer_chunk_header,
                       char*              qstring,
                       int                stride,
                       int                qstr_offset,
                       int                bs,
                       int                chunk_size,
                       int                max_chunk_base_num,
                       int                trim_head_base_num,
                       int                trim_tail_base_num)
{
  if (bs <= 0)
    return 0;

  if (trim_head_base_num + trim_tail_base_num > max_chunk_base_num)
    return -2;

  const bool do_trim = (max_chunk_base_num != 0 && trim_head_base_num != 0 &&
                        trim_tail_base_num != 0);

  if (do_trim)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int idx = 0; idx < bs; idx++)
    {
      process_one_with_trim(&buffer_chunk_header[idx],
                            qstring + idx * qstr_offset, qstr_offset,
                            chunk_size, max_chunk_base_num, trim_head_base_num,
                            trim_tail_base_num, stride);
    }
  }
  else
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int idx = 0; idx < bs; idx++)
    {
      process_one_no_trim(&buffer_chunk_header[idx],
                          qstring + idx * qstr_offset, qstr_offset, chunk_size,
                          stride);
    }
  }

  return 0;
}
