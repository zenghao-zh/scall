/*******************************************************************************
 * FILENAME:      kernel_cpu.h
 *
 * CPU port of kernel.cu — replaces all __device__ / __global__ functions
 * with plain C++ equivalents.
 *******************************************************************************/

#pragma once

#include "rta_postprocess_common.h"

// Main entry point — identical signature to the CUDA version in kernel.h,
// but operates entirely on host memory.
int native_postprocess(BufferChunkHeader* buffer_chunk_header,
                       char*              qstring,
                       int                stride,
                       int                qstr_offset,
                       int                bs,
                       int                chunk_size,
                       int                max_chunk_base_num,
                       int                trim_head_base_num,
                       int                trim_tail_base_num);
