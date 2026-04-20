/*******************************************************************************
 * FILENAME:      rta_postprocess_common.h
 *
 * AUTHORS:       Pan Shaohua
 *
 * START DATE:    2023-12-08 11:13:39
 *
 * Last Modified: 2025-11-25 18:09:55
 *
 * CONTACT:       panshaohua@genomics.cn
 *******************************************************************************/

#ifndef RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_COMMON_H_
#define RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_COMMON_H_

#include <cstdint>

enum STATUS_CODE
{
  CCA_RTA_POSTPROCESS_RET_OK              = 0,  // 正常返回
  CCA_RTA_POSTPROCESS_RET_INIT_FAILED     = 1,  // 初始化失败
  CCA_RTA_POSTPROCESS_RET_NULL_PTR        = 2,  // 输入是空指针
  CCA_RTA_POSTPROCESS_RET_INTERNAL_ERROR  = 3,  // 内部处理错误
  CCA_RTA_POSTPROCESS_RET_DATA_TYPE_ERROR = 4,  // 数据类型错误
};

enum DEVICE_TYPE
{
  CPU = 0,
  GPU = 1
};

typedef struct RTAPostprocessParams
{
  int batch_size    = 256;
  int chunk_size    = 5000;
  int qstring_len   = 1000;
  int header_offset = 15;
  RTAPostprocessParams()
  {
    batch_size    = 256;
    chunk_size    = 5000;
    qstring_len   = 1000;
    header_offset = 15;
  }
} RTAPostprocessParams;

typedef struct BufferChunkHeader
{
  int chip_num   = 0;  //芯片编号
  int chn_num    = 0;  // 1~1024
  int chn_status = 1;  // 0: turn off, 1: turn on
  int cycle_num  = 0;  //周期编号
  int read_num   = 0;  // reads的序号
  int chunk_num  = 0;  // chunk序号

  int sensor_num     = 0;
  int chunk_base_num = 0;
  int chunk_type     = 0;
  int trim_err_type  = 0;

  float openpore_before_median = 0.0f;
  float openpore_before_std    = 0.0f;
  float chunk_err              = 0;

  int64_t overlap = 0;  // overlap数据的长度
  int64_t offset =
    0;  // if offset is not zero, means read has not pass through the pore
  int64_t start_idx = 0;

  BufferChunkHeader()
  {
    chip_num               = 0;
    chn_num                = 0;
    chn_status             = 1;
    cycle_num              = 0;
    read_num               = 0;
    chunk_num              = 0;
    overlap                = 0;
    offset                 = 0;
    start_idx              = 0;
    openpore_before_median = 0;
    openpore_before_std    = 0;
    sensor_num             = 0;
    chunk_err              = 0;
    chunk_base_num         = 0;
    chunk_type             = 0;
    trim_err_type          = 0;
  }
} BufferChunkHeader;

#endif  // RTA_POSTPROCESS_INCLUDE_RTA_POSTPROCESS_COMMON_H_
