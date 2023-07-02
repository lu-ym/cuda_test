#ifndef UTILSM_H__ // UTILSM_H__
#define UTILSM_H__ // UTILSM_H__
#include <iomanip>
#include <iostream>


// clock per millisecond
#define CLOCK_PER_MILLSEC (CLOCKS_PER_SEC / 1000)

unsigned long long get_time_us(void);

__inline void print_time_us(unsigned long long time_d) {
  std::cout << std::setw(6) << time_d / 1000 << " ms " << std::setw(3)
            << time_d % 1000 << " us" << std::endl;
}

template <typename TT>
void print_vector(TT* const vector_l, const unsigned int start_n,
                  const unsigned int len) {
  unsigned int start = start_n;
  for (; start < len; start++) {
    if (!(start % 16)) {
      if (start != start_n) std::cout << std::endl;
      std::cout << std::setw(6) << start << ": " << std::flush;
    }
    std::cout << std::setw(8) << vector_l[start] << std::flush;
  }
  std::cout << std::endl;
}

inline bool pointEqual(float a, float b) {
  // return (std::abs(a - b) > 0.001000)? false:true;
  return (std::abs(a - b) / std::abs(a) > 0.000005) ? false : true;
}

#define KB_SIZE 1024
#define MB_SIZE (KB_SIZE * 1024)
template <typename T>
void bandwidth_print(unsigned long long duration, T catgr, unsigned int len) {
  unsigned long long len_total = sizeof(catgr) * len;
  double bandwidth = (double)len_total * 1000000 / duration;  // TODO: overflow?
  std::cout << "  Bandwidth is:           ";
  if (bandwidth > (double)MB_SIZE) {
    std::cout << std::fixed << std::setprecision(3) << bandwidth / MB_SIZE
              << " MB/s" << std::endl;
  } else if (bandwidth > (double)KB_SIZE) {
    std::cout << std::fixed << std::setprecision(3) << bandwidth / KB_SIZE
              << " KB/s" << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(3) << bandwidth << " B/s"
              << std::endl;
  }
  std::cout << "    Data size is:              ";
  if (len_total > (double)MB_SIZE) {
    std::cout << len_total / MB_SIZE << " MB" << std::endl;
  } else if (len_total > (double)KB_SIZE) {
    std::cout << len_total / KB_SIZE << " KB" << std::endl;
  } else {
    std::cout << len_total << " bytes" << std::endl;
  }
  std::cout << "    duration is:          ";
  print_time_us(duration);
  return;
}

#endif // UTILSM_H__
