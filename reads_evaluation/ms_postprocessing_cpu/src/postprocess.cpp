/*******************************************************************************
 * FILENAME:      postprocess.cpp
 *
 * CPU port — all CUDA memory management and memcpy calls removed.
 * Calls native_postprocess() directly on host pointers.
 *******************************************************************************/

#include "postprocess.h"
#include "kernel_cpu.h"
#include "spdlog/spdlog.h"

namespace CCA
{

RTAPostProcess::RTAPostProcess(const RTAPostprocessParams& postprocess_params,
                               int /*gpu_id*/)
{
  batch_size_    = postprocess_params.batch_size;
  chunk_size_    = postprocess_params.chunk_size;
  qstring_len_   = postprocess_params.qstring_len;
  header_offset_ = postprocess_params.header_offset;

  buffer_chunk_header_ = new BufferChunkHeader[batch_size_];
}

RTAPostProcess::~RTAPostProcess()
{
  delete[] buffer_chunk_header_;
}

STATUS_CODE RTAPostProcess::Process(BufferChunk* buffer_chunk,
                                    char*        qs_code,
                                    int          stride,
                                    int          max_chunk_base_num,
                                    int          trim_head_base_num,
                                    int          trim_tail_base_num)
{
  // Copy fields from BufferChunk → BufferChunkHeader (same as GPU version)
  for (int i = 0; i < batch_size_; i++)
  {
    buffer_chunk_header_[i].chip_num               = buffer_chunk[i].chip_num;
    buffer_chunk_header_[i].chn_num                = buffer_chunk[i].chn_num;
    buffer_chunk_header_[i].chn_status             = buffer_chunk[i].chn_status;
    buffer_chunk_header_[i].cycle_num              = buffer_chunk[i].cycle_num;
    buffer_chunk_header_[i].read_num               = buffer_chunk[i].read_num;
    buffer_chunk_header_[i].chunk_num              = buffer_chunk[i].chunk_num;
    buffer_chunk_header_[i].overlap                = buffer_chunk[i].overlap;
    buffer_chunk_header_[i].offset                 = buffer_chunk[i].offset;
    buffer_chunk_header_[i].start_idx              = buffer_chunk[i].start_idx;
    buffer_chunk_header_[i].openpore_before_median = buffer_chunk[i].openpore_before_median;
    buffer_chunk_header_[i].openpore_before_std    = buffer_chunk[i].openpore_before_std;
    buffer_chunk_header_[i].sensor_num             = buffer_chunk[i].sensor_num;
    // outputs — reset before processing
    buffer_chunk_header_[i].chunk_err      = 0;
    buffer_chunk_header_[i].chunk_base_num = 0;
    buffer_chunk_header_[i].trim_err_type  = 0;
  }

  int rtn = native_postprocess(buffer_chunk_header_, qs_code, stride,
                               qstring_len_, batch_size_, chunk_size_,
                               max_chunk_base_num, trim_head_base_num,
                               trim_tail_base_num);
  if (rtn == -2)
  {
    SPDLOG_LOGGER_ERROR(spdlog::default_logger(),
                        "trim_head_base_num + trim_tail_base_num > max_chunk_base_num");
    return CCA_RTA_POSTPROCESS_RET_INTERNAL_ERROR;
  }

  // Copy outputs back to caller's BufferChunk array
  for (int i = 0; i < batch_size_; i++)
  {
    buffer_chunk[i].chunk_err      = buffer_chunk_header_[i].chunk_err;
    buffer_chunk[i].chunk_base_num = buffer_chunk_header_[i].chunk_base_num;
    buffer_chunk[i].trim_err_type  = buffer_chunk_header_[i].trim_err_type;
  }

  return CCA_RTA_POSTPROCESS_RET_OK;
}

void RTAPostProcess::SetErrorCode(STATUS_CODE status_code)
{
  status_code_ = status_code;
}

}  // namespace CCA
