/*******************************************************************************
 * FILENAME:      rta_common.h
 *
 * AUTHORS:       Pan Shaohua
 *
 * START DATE:    2023-10-24 16:37:45
 *
 * Last Modified: 2025-11-25 18:09:50
 *
 * CONTACT:       panshaohua@genomics.cn
 *******************************************************************************/

#ifndef READSALIGNMENT_INCLUDE_RTA_COMMON_H_
#define READSALIGNMENT_INCLUDE_RTA_COMMON_H_

#include <cstdint>
#include <cstring>
#include <vector>

/// 数据块的头部信息
typedef struct BatchFrameHeader
{
  int              chip_num   = 1;  // 芯片编号，从1开始
  int              cycle_num  = 1;  // 周期编号，从1开始
  int              frame_num  = 1;  // 帧号 从1开始
  int              seq_status = 1;  // 0: stop, 1: sequencing
  std::vector<int> chn_status;
  std::vector<int> sensor_status;

  explicit BatchFrameHeader(int chn_num = 1024)
  {
    chn_status.resize(chn_num, 1);
    sensor_status.resize(chn_num, 1);
  }

} BatchFrameHeader;

/// <summary>
/// 预处理返回数据结构
/// </summary>
typedef struct BufferChunk
{
  int     chip_num   = 0;  // 芯片编号
  int     chn_num    = 0;  // 1~1024
  int     chn_status = 1;  // 0: turn off, 1: turn on
  int     cycle_num  = 0;  // 周期编号
  int     read_num   = 0;  // reads的序号
  int     chunk_num  = 0;  // chunk序号
  int64_t overlap    = 0;  // overlap数据的长度
  int64_t offset =
    0;  // if offset is not zero, means read has not pass through the pore
  int64_t start_idx              = 0;
  float   openpore_before_median = 0.0f;
  float   openpore_before_std    = 0.0f;
  float   read_median            = 0.0f;  // median of the chunk header
  float   read_std               = 0.0f;  // std of the chunk header
  int     sensor_num             = 0;
  float   chunk_err              = 0;
  int     chunk_base_num         = 0;
  int     chunk_type = 0;  // 0 means others, 1 means reads, 2 means adaptor
  int trim_err_type  = 0;  // 表示切除头部或者尾部chunk个数的类型
  float buffer[5000]{};

  void reset()
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
    read_median            = 0;  // median of the chunk header
    read_std               = 0;  // std of the chunk header
    sensor_num             = 0;
    chunk_err              = 0;
    chunk_base_num         = 0;
    chunk_type             = 0;
    trim_err_type          = 0;
    memset(buffer, 0, sizeof(float) * 5000);
  }

} BufferChunk;

// reads设置
struct ReadsSetInfo
{
  int   min_read_len          = 300;   // 最小reads长度
  int   overlap               = 500;   // overlap长度
  int   frame_len             = 5000;  // 帧长度
  int   cut_len               = 50;
  int   cut_offset            = 5;     // 计算openpore mean std
  int   total_chn_num         = 1024;  // 总通道数
  int   read_type_threshold   = 1000;
  float min_read_std          = 0.0f;
  float min_adaptor_std       = 20.0f;
  float openpore_median_thres = 180;
  float openpore_std_thres    = 8;
  float init_openpore_val     = 350;
  float default_openpore_val  = 220.0f;
  float default_openpore_std  = 1.0f;
  float off_mean_max          = 10.0f;  //  filter the pores off
};

#endif  // READSALIGNMENT_INCLUDE_RTA_COMMON_H_
