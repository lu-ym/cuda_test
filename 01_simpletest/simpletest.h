#ifndef SIMPLETEST_H__
#define SIMPLETEST_H__

// the thread is organized as one-dimensional
struct __align__(32) IDS {
  unsigned short tIdx;
  // unsigned short tIdy;
  unsigned short bIdx;
  // unsigned short bIdy;
};
void gpu_test(int block_size, int grid_sz, struct IDS* const globalMem,
              const unsigned short len);

#endif
