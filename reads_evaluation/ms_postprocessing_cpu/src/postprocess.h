/*******************************************************************************
 * FILENAME:      postprocess.h
 *
 * CPU port — removed all CUDA device-memory members.
 * Public interface is identical to the original GPU version.
 *******************************************************************************/

#pragma once

#include "rta_common.h"
#include "rta_postprocess_common.h"

namespace CCA
{
class RTAPostProcess
{
public:
  explicit RTAPostProcess(const RTAPostprocessParams& postprocess_params,
                          int                         gpu_id);
  ~RTAPostProcess();

  [[nodiscard]] STATUS_CODE Process(BufferChunk* buffer_chunk,
                                    char*        qscode,
                                    int          stride,
                                    int          max_chunk_base_num,
                                    int          trim_head_base_num,
                                    int          trim_tail_base_num);

  void SetErrorCode(STATUS_CODE status_code);

private:
  int                batch_size_;
  int                chunk_size_;
  int                qstring_len_;
  int                header_offset_;
  BufferChunkHeader* buffer_chunk_header_;  // CPU-only, no GPU mirror

  STATUS_CODE status_code_{CCA_RTA_POSTPROCESS_RET_OK};
};

}  // namespace CCA
