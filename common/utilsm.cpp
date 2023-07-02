#include "utilsm.h"

#if defined(_WIN32)
#include <windows.h>
// #elif defined(linux)
// #include <time.h>
#else
#include <ctime>
#endif  // Microsoft windows
/**
 * \brief Get the time us object.
 *
 * \return double. Unit us. Accuracy is according to system function.
 */
unsigned long long get_time_us(void) {
// #if defined(_WIN32)
#ifdef _WIN32
  FILETIME ft;
  unsigned long long time;
  // double time_ms;
  GetSystemTimePreciseAsFileTime(&ft);  // unit: 100ns
  time = (unsigned long long)ft.dwHighDateTime << 32;
  time |= ft.dwLowDateTime;
  return time / 10;
// #elif defined(linux)
// int gettimeofday ( struct timeval * tv , struct timezone * tz );  // unit:1us
#else
  // CLOCKS_PER_SEC is 1000 on windows and 10^6 on Linux.
  return std::clock() * (1000000 / CLOCKS_PER_SEC);
#endif
}



