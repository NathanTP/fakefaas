#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include <cmath>

class SortState {
  public:
    SortState(unsigned int *in, size_t len);
    ~SortState();

    // Sort 'width' bits of d_in into d_out starting at bit position 'offset'.
    // Steps must be aligned to 2-bit boundaries (width and offset must be
    // multiples of 2).
    void Step(int offset, int width);

    // Returns the start point in the partially sorted array of each group of
    // [offset, offset+width) bits. You must have used Step to sort this group
    // (multiple steps are fine so long as the entire group has been covered).
    void GetBoundaries(unsigned int *boundaries, int offset, int width);

    // Copy the (potentially partially) sorted result to out. At least one call
    // to Step must be made before calling GetResult.
    void GetResult(unsigned int *out);

  private:
      // Input and output device pointers
      unsigned int *d_out;
      unsigned int *d_in;

      // The per-block, per-bit prefix sums (where this value goes in the per-block 2bit group)
      unsigned int *d_prefix_sums;

      // per-block starting index (count) of each 2bit grouped by 2bit (d_block_sums[0-nblock] are all the 0 2bits)
      unsigned int *d_block_sums;

      // prefix-sum of d_block_sums, e.g. the starting position for each block's 2bit group
      // (d_scan_block_sums[1] is where block 1's 2bit group 0 should start)
      unsigned int *d_scan_block_sums;

      unsigned int data_len;
      unsigned int block_sz;
      unsigned int grid_sz;
      unsigned int shmem_sz;
      unsigned int prefix_sums_len;
      unsigned int block_sums_len;
      unsigned int scan_block_sums_len;
};

#endif
