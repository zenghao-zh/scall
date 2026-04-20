/*******************************************************************************
 * FILENAME:      npy.h  (CPU-only port)
 *
 * Original npy.h had CUDA paths for device_type == 1.
 * This version keeps only the host-memory path (device_type == 0 / memcpy).
 * The loadNpy / writeNpy API is unchanged so sample code compiles as-is.
 *******************************************************************************/

#ifndef CYCLONEACC_SAMPLES_NPY_H_
#define CYCLONEACC_SAMPLES_NPY_H_

#include <cassert>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace CycloneAcc
{
namespace npy
{
  namespace internal
  {
    template <class T> struct is_complex : std::false_type {};
    template <class T> struct is_complex<std::complex<T>> : std::true_type {};

    template <class T, class Enable = void> struct map_type { const static char value = '?'; };
    template <class T> struct map_type<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
    { const static char value = 'f'; };
    template <class T> struct map_type<T, typename std::enable_if<std::is_signed<T>::value && std::is_integral<T>::value>::type>
    { const static char value = 'i'; };
    template <class T> struct map_type<T, typename std::enable_if<std::is_unsigned<T>::value && std::is_integral<T>::value>::type>
    { const static char value = 'u'; };
    template <> struct map_type<bool, void> { const static char value = 'b'; };
    template <class T> struct map_type<T, typename std::enable_if<is_complex<T>::value>::type>
    { const static char value = 'c'; };

    inline std::string map_endian()
    {
      int x = 1;
      return (reinterpret_cast<char*>(&x)[0]) ? "<" : ">";
    }

    inline void parse_header(FILE*                fp,
                             size_t&              word_size,
                             std::vector<size_t>& shape,
                             bool&                fortran_order)
    {
      const int leading_size = 11;
      char      buffer[256];
      if (fread(buffer, sizeof(char), leading_size, fp) != static_cast<size_t>(leading_size))
        throw std::runtime_error("parse_header: failed fread");

      std::string header = fgets(buffer, 256, fp);
      assert(header[header.size() - 1] == '\n');

      size_t start, end;

      start = header.find("fortran_order");
      if (start == std::string::npos)
        throw std::runtime_error("parse_header: missing 'fortran_order'");
      start += 16;
      fortran_order = (header.substr(start, 4) == "True");

      start = header.find("(");
      end   = header.find(")");
      if (start == std::string::npos || end == std::string::npos)
        throw std::runtime_error("parse_header: missing '(' or ')'");

      std::regex  num_regex("[0-9][0-9]*");
      std::smatch sm;
      shape.clear();
      std::string str_shape = header.substr(start + 1, end - start - 1);
      while (std::regex_search(str_shape, sm, num_regex))
      {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
      }

      start = header.find("descr");
      if (start == std::string::npos)
        throw std::runtime_error("parse_header: missing 'descr'");
      start += 9;

      std::string str_ws = header.substr(start + 2);
      end                = str_ws.find("'");
      word_size          = static_cast<size_t>(atoi(str_ws.substr(0, end).c_str()));
    }

    template <typename T>
    std::vector<char> create_header(const std::vector<size_t>& shape)
    {
      std::string dict;
      dict += "{'descr': '" + map_endian() + map_type<T>::value +
              std::to_string(sizeof(T)) + "', ";
      dict += "'fortran_order': False, ";
      dict += "'shape': (";
      dict += std::to_string(shape[0]);
      for (size_t i = 1; i < shape.size(); i++) { dict += ", "; dict += std::to_string(shape[i]); }
      if (shape.size() == 1) dict += ",";
      dict += "), }";

      int remainder = 16 - (10 + static_cast<int>(dict.size())) % 16;
      dict.insert(dict.end(), remainder, ' ');
      dict.back() = '\n';

      std::string header;
      header += static_cast<char>(0x93);
      header += "NUMPY";
      header += static_cast<char>(0x01);
      header += static_cast<char>(0x00);
      auto size = static_cast<uint16_t>(dict.size());
      header += static_cast<char>(reinterpret_cast<char*>(&size)[0]);
      header += static_cast<char>(reinterpret_cast<char*>(&size)[1]);
      header.insert(header.end(), dict.begin(), dict.end());
      return std::vector<char>(header.begin(), header.end());
    }
  }  // namespace internal

  class Array
  {
  public:
    Array(const std::vector<size_t>& shape, size_t word_size, bool fortran_order)
      : mShape(shape), mWordSize(word_size), mFortranOrder(fortran_order)
    {
      mNumValues = 1;
      for (auto s : mShape) mNumValues *= s;
      mDataHolder = std::make_shared<std::vector<char>>(mNumValues * mWordSize);
    }
    Array() : mShape(0), mWordSize(0), mNumValues(0), mFortranOrder(false) {}

    template <typename T>       T* data()       { return reinterpret_cast<T*>(mDataHolder->data()); }
    template <typename T> const T* data() const { return reinterpret_cast<const T*>(mDataHolder->data()); }
    template <typename T> std::vector<T> as_vector() const
    {
      const T* p = data<T>();
      return std::vector<T>(p, p + mNumValues);
    }
    size_t num_bytes()  const { return mDataHolder->size(); }
    size_t num_values() const { return mNumValues; }
    std::vector<size_t> shape() const { return mShape; }

  private:
    std::shared_ptr<std::vector<char>> mDataHolder;
    std::vector<size_t>                mShape;
    size_t                             mWordSize;
    size_t                             mNumValues;
    bool                               mFortranOrder;
  };

  static Array load(const std::string& fname)
  {
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("load: cannot open " + fname);
    std::vector<size_t> shape;
    size_t word_size; bool fortran_order;
    internal::parse_header(fp, word_size, shape, fortran_order);
    Array arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    fclose(fp);
    if (nread != arr.num_bytes()) throw std::runtime_error("load: short read");
    return arr;
  }

  template <typename T>
  static void save(const std::string& fname, const T* data, const std::vector<size_t>& shape)
  {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp) throw std::runtime_error("save: cannot open " + fname);
    auto header = internal::create_header<T>(shape);
    fwrite(header.data(), sizeof(char), header.size(), fp);
    size_t nels = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
  }
}  // namespace npy
}  // namespace CycloneAcc

// ─────────────────────────────────────────────────────────────────────────────
// Public loadNpy / writeNpy helpers — CPU-only (device_type==0 paths only)
// ─────────────────────────────────────────────────────────────────────────────

namespace CycloneAcc
{

// float array, host target
inline void loadNpy(const std::string& npy_path,
                    void**             host_vec,
                    int                vec_size,
                    int                /*device_type*/)
{
  auto               input    = CycloneAcc::npy::load(npy_path);
  std::vector<float> host_buf = input.as_vector<float>();
  if (vec_size != static_cast<int>(host_buf.size()))
    throw std::runtime_error("loadNpy: size mismatch for " + npy_path);
  std::memcpy(*host_vec, host_buf.data(), vec_size * sizeof(float));
  std::cout << "[CycloneAcc] load npy " << npy_path
            << ", size: " << vec_size << std::endl;
}

// typed array, host target
template <class T>
void loadNpy(const std::string& npy_path,
             void**             host_vec,
             size_t             vec_size,
             int                /*device_type*/)
{
  auto           input    = CycloneAcc::npy::load(npy_path);
  std::vector<T> host_buf = input.as_vector<T>();
  if (vec_size != host_buf.size())
    throw std::runtime_error("loadNpy: size mismatch for " + npy_path);
  std::memcpy(*host_vec, host_buf.data(), vec_size * sizeof(T));
  std::cout << "[CycloneAcc] load npy " << npy_path
            << ", size: " << vec_size << std::endl;
}

// writeNpy — host source
template <class T>
void writeNpy(const std::string&         npy_path,
              const void*                host_vec,
              const std::vector<size_t>& shape,
              int                        /*device_type*/)
{
  size_t vec_size = std::accumulate(shape.begin(), shape.end(), size_t(1),
                                    std::multiplies<size_t>());
  CycloneAcc::npy::save(npy_path, reinterpret_cast<const T*>(host_vec), shape);
  std::cout << "[CycloneAcc] save npy " << npy_path
            << ", size: " << vec_size << std::endl;
}

}  // namespace CycloneAcc

#endif  // CYCLONEACC_SAMPLES_NPY_H_
