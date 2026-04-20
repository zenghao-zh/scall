/*******************************************************************************
 * FILENAME:      rta_postprocess.h
 *
 * AUTHORS:       Pan Shaohua
 *
 * START DATE:    2023-12-08 11:07:41
 *
 * Last Modified: 2025-11-25 18:09:57
 *
 * CONTACT:       panshaohua@genomics.cn
 *******************************************************************************/

#ifndef RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_H_
#define RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_H_

#include "rta_common.h"
#include "rta_postprocess_common.h"

namespace CCA
{
STATUS_CODE
Init(void** handle, RTAPostprocessParams& postprocess_params, int gpu_id);

/**
 * Process interface
 * @param handle: [IN]
 * @param buffer_chunk: [IN], size: batch_size * 5000,
 * @param qscode: [IN], size: batch_size * 1000,
 * @param max_chunk_base_num: [IN],
 * 当前chunk的base长度大于max_chunk_base_num，需要切除头部base计算chunk_err
 * @param trim_head_base_num: [IN]  计算chunk_err时需要切除头部长度
 * @param trim_tail_base_num: [IN]  计算chunk_err时需要切除尾部长度
 * @param stride: [IN]
 * @return true if successful else false
 * @note 模块主函数
 */
STATUS_CODE PostProcess(void*        handle,
                        BufferChunk* buffer_chunk,
                        char*        qscode,
                        int          max_chunk_base_num = 100,
                        int          trim_head_base_num = 40,
                        int          trim_tail_base_num = 40,
                        int          stride             = 5);

}  // namespace CCA

#endif  // RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_H_
