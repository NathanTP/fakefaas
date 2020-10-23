#include "sort.h"
#include <stdio.h>
#include <string.h>

#define MAX_BLOCK_SZ 128
/* #define MAX_BLOCK_SZ 256 */

// Return bits val[pos:width)
#define group_bits(val, pos, width) ((val >> pos) & ((1 << width) - 1));

// Detect the boundaries of each '[offset, offset+group_width)" group in d_in
// and place in d_boundaries. d_in is assumed to be already sorted by the group
// bits. d_boundaries must be 2^group_width long.
__global__ void gpu_groups(unsigned int* d_boundaries, unsigned int* d_in, int offset, int group_width, unsigned int d_in_len)
{
  unsigned int th_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(th_idx < d_in_len) {
    unsigned int prev_idx = (th_idx == 0) ? 0 : th_idx - 1;
    unsigned int th_group = group_bits(d_in[th_idx], offset, group_width);
    unsigned int prev_group = group_bits(d_in[prev_idx], offset, group_width);

    if(th_group != prev_group) {
      d_boundaries[th_group] = th_idx;
    }
  }
}

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // need shared memory array for:
    // - block's share of the input data (local sort will be put here too)
    // - mask outputs
    // - scanned mask outputs
    // - merged scaned mask outputs ("local prefix sum")
    // - local sums of scanned mask outputs
    // - scanned local sums of scanned mask outputs

    // for all radix combinations:
    //  build mask output for current radix combination
    //  scan mask ouput
    //  store needed value from current prefix sum array to merged prefix sum array
    //  store total sum of mask output (obtained from scan) to global block sum array
    // calculate local sorted address from local prefix sum and scanned mask output's total sums
    // shuffle input block according to calculated local sorted addresses
    // shuffle local prefix sums according to calculated local sorted addresses
    // copy locally sorted array back to global memory
    // copy local prefix sum array back to global memory

    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    // s_mask_out[] will be scanned in place
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    // 2bit-specific prefix-sum for each elem (e.g. where in the 2bit's output block this elem should go)
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    // per-block per-2bit count (how many elems of each 2bit there are)
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    // per-block starting point for each 2bit
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];

    unsigned int thid = threadIdx.x;

    // Copy block's portion of global input data to shared memory
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;

    __syncthreads();

    //  To extract the correct 2 bits, we first shift the number
    //  to the right until the correct 2 bits are in the 2 LSBs,
    //  then mask on the number with 11 (3) to remove the bits
    //  on the left
    unsigned int t_data = s_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (unsigned int i = 0; i < 4; ++i)
    {
        // Zero out s_mask_out
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele)
        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        // Shift elements to produce the same effect as exclusive scan
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    if (thid == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;
        
        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];
    }
    //XXX d_out_sorted is sorted per block
    //XXX d_prefix_sums is the per-block, per-bit prefix sums
    //XXX s_scan_mask_out_sums has the per-block starting index of each 2bit. It is stored in d_block_sums.
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

bool check(unsigned int* d_in, unsigned int* d_prefix_sums, unsigned int len, int shift_width)
{
    int nprefix = (1 << (shift_width + 2));
    unsigned int *h_dat = new unsigned int[len];
    unsigned int *h_prefix_sums = new unsigned int[len];
    unsigned int *prefix_boundaries = new unsigned int[nprefix];
    cudaMemcpy(h_dat, d_in, sizeof(unsigned int)*len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prefix_sums, d_prefix_sums, sizeof(unsigned int)*len, cudaMemcpyDeviceToHost);

    unsigned int old_prefix = 0;
    prefix_boundaries[0] = 0;
    bool success = true;
    int nchange = 0;

    for(unsigned int i = 0; i < len; i++) {
        // Grab total prefix sorted so far
        unsigned int prefix = h_dat[i] & (nprefix - 1);
        if(prefix < old_prefix) {
            printf("prefix changed from %d to %d at %d\n", old_prefix, prefix, i);
            std::cout << "Prefixes not increasing monotonically!\n";
            success = false;
            break;
        }
        if(prefix != old_prefix) {
            nchange++;
            if(prefix > (unsigned int)(nprefix - 1)) {
                printf("Prefix (%d) out of range (expected < %d prefixes)\n", prefix, nprefix);
                break;
            }
            prefix_boundaries[prefix] = i;
        }
        old_prefix = prefix;
    }
    printf("nchange=%d\n", nchange);
    if(success) {
        for (int i = 0; i < nprefix; i++) {
            printf("Prefix %d at %u\n", i, prefix_boundaries[i]);
        }
    }
//    printf("Prefix sums:\n");
//    for(unsigned int i = 0; i < len; i++) {
//        printf("%u:\t%u\t(%x)\n", i, h_prefix_sums[i], h_dat[i]);
//    }
    delete[] h_dat;
    delete[] prefix_boundaries;
    return success;
}

// Allocate all intermediate state needed to perform a sort of d_in into d_out
SortState::SortState(unsigned int* in, size_t len) : data_len(len)
{
    block_sz = MAX_BLOCK_SZ;
    grid_sz = data_len / block_sz;

    //grid_sz was the floor, add an extra block if there was extra data
    if (data_len % block_sz != 0)
        grid_sz += 1;

    checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * data_len));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * data_len));
    checkCudaErrors(cudaMemcpy(d_in, in, sizeof(unsigned int) * data_len, cudaMemcpyHostToDevice));

    // The per-block, per-bit prefix sums (where this value goes in the per-block 2bit group)
    prefix_sums_len = data_len;
    checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * prefix_sums_len));
    checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * prefix_sums_len));

    // per-block starting index (count) of each 2bit grouped by 2bit (d_block_sums[0-nblock] are all the 0 2bits)
    // e.g. 4 indices per block
    block_sums_len = 4 * grid_sz;
    checkCudaErrors(cudaMalloc(&(d_block_sums), sizeof(unsigned int) * block_sums_len));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * block_sums_len));

    // prefix-sum of d_block_sums, e.g. the starting position for each block's 2bit group
    // (d_scan_block_sums[1] is where block 1's 2bit group 0 should start)
    scan_block_sums_len = block_sums_len;
    checkCudaErrors(cudaMalloc(&(d_scan_block_sums), sizeof(unsigned int) * block_sums_len));
    checkCudaErrors(cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * block_sums_len));

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = block_sz;
    unsigned int s_mask_out_len = block_sz + 1;
    unsigned int s_merged_scan_mask_out_len = block_sz;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    shmem_sz = (s_data_len 
                            + s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);
}

// Destroy's everything allocated by init_sort(). It is invalid to use state
// after calling destroy_state(state). Noteably, this does not deallocate
// state->d_in or d_out, you must free those independently.
SortState::~SortState()
{
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_scan_block_sums));
    checkCudaErrors(cudaFree(d_block_sums));
    checkCudaErrors(cudaFree(d_prefix_sums));

}

void SortState::Step(int offset, int width) {
    for (int shift_width = offset; shift_width < offset + width; shift_width += 2)
    {
        // per-block sort. Also creates blockwise prefix sums.
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, 
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                data_len, 
                                                                block_sz);

        // create global prefix sum arrays
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, 
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    data_len, 
                                                    block_sz);
    }
}
 
// A fallback CPU-only boundary detection
/* void SortState::GetBoundaries(unsigned int *boundaries, int offset, int width) { */
/*     auto out = new unsigned int[data_len]; */
/*     checkCudaErrors(cudaMemcpy(out, d_in, sizeof(unsigned int) * data_len, cudaMemcpyDeviceToHost)); */
/*  */
/*     boundaries[0] = 0; */
/*     unsigned int curGroup = 0; */
/*     for(unsigned int i = 1; i < data_len; i++) { */
/*       unsigned int bits = group_bits(out[i], offset, width); */
/*       if(bits != curGroup) { */
/*         for(unsigned int j = 1; j <= (bits - curGroup); j++) { */
/*             boundaries[curGroup + j] = i; */
/*         } */
/*         curGroup = bits; */
/*       } */
/*     } */
/*     delete[] out; */
/* } */

void SortState::GetBoundaries(unsigned int *boundaries, int offset, int width) {
    int nboundary = (1 << width);
    unsigned int *d_boundaries;
    checkCudaErrors(cudaMalloc(&d_boundaries, nboundary*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_boundaries, 0, nboundary*sizeof(unsigned int)));

    gpu_groups<<<grid_sz, block_sz>>>(d_boundaries, d_in, offset, width, data_len);

    checkCudaErrors(cudaMemcpy(boundaries, d_boundaries, sizeof(unsigned int) * nboundary, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_boundaries));

    // Empty groups can't be detected by gpu_groups() so we have to fill them
    // in here. nboundaries is assumed to be small so not worth using the GPU
    // for.
    int prev = data_len;
    for(int group = nboundary - 1; group > 1; group--) {
      if(boundaries[group] == 0) {
        boundaries[group] = prev;
      }
      prev = boundaries[group];
    }

    /* for(int group = 1; group < nboundary; group++) { */
    /*   if(boundaries[group] == 0) { */
    /*     boundaries[group] = boundaries[group - 1]; */
    /*   } */
    /* } */
}

void SortState::GetResult(unsigned int *out) {
    checkCudaErrors(cudaMemcpy(out, d_in, sizeof(unsigned int) * data_len, cudaMemcpyDeviceToHost));
}
