/*******************************************************************************
 * FILENAME:      rta_postprocess.cpp
 *
 * CPU port — gpu_id parameter accepted for API compatibility but not used.
 *******************************************************************************/

#include "rta_postprocess.h"
#include "postprocess.h"
#include "spdlog/spdlog.h"

#include "version.h"

namespace CCA
{

STATUS_CODE
Init(void** handle, RTAPostprocessParams& postprocess_params, int /*gpu_id*/)
{
  spdlog::info("MSPostprocessVersion (CPU): {}", MSPOSTPROCESS_VERSION);

  auto rta_postprocess = new RTAPostProcess(postprocess_params, 0);
  *handle              = rta_postprocess;

  if (!(*handle))
    return CCA_RTA_POSTPROCESS_RET_INIT_FAILED;

  SPDLOG_LOGGER_INFO(spdlog::default_logger(),
                     "Initialize Postprocess (CPU) Success !!!");
  return CCA_RTA_POSTPROCESS_RET_OK;
}

STATUS_CODE PostProcess(void*        handle,
                        BufferChunk* buffer_chunk,
                        char*        qscode,
                        int          max_chunk_base_num,
                        int          trim_head_base_num,
                        int          trim_tail_base_num,
                        int          stride)
{
  auto rta_postprocess = static_cast<RTAPostProcess*>(handle);
  if (!rta_postprocess)
  {
    SPDLOG_LOGGER_ERROR(spdlog::default_logger(),
                        "please call Init first!");
    return CCA_RTA_POSTPROCESS_RET_NULL_PTR;
  }

  if (!buffer_chunk || !qscode)
  {
    SPDLOG_LOGGER_ERROR(spdlog::default_logger(),
                        "input data is null");
    rta_postprocess->SetErrorCode(CCA_RTA_POSTPROCESS_RET_NULL_PTR);
    return CCA_RTA_POSTPROCESS_RET_NULL_PTR;
  }

  return rta_postprocess->Process(buffer_chunk, qscode, stride,
                                  max_chunk_base_num, trim_head_base_num,
                                  trim_tail_base_num);
}

}  // namespace CCA
