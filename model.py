"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] *= scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float max_val = matrix[r * cols];

        for (int c = 1; c < cols; ++c) {
            max_val = fmaxf(max_val, matrix[r * cols + c]);
        }

        out[r] = max_val;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c) {
            sum += matrix[r * cols + c];
        }

        out[r] = sum;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float result = 0.0f;

    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;

        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }

        c[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        const float* q_row = q + i * head_dim;
        const float* k_row = k + j * head_dim;

        float dot = dot_product(q_row, k_row, head_dim);

        scores[i * seq_len + j] = dot / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x;

    if (r >= rows) {
        return;
    }

    __shared__ float shared_max[1024];
    __shared__ float shared_sum[1024];

    int tid = threadIdx.x;

    // Find the maximum value in this row.
    float local_max = -3.402823466e+38F;

    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, matrix[r * cols + c]);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce maximum across the block.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(
                shared_max[tid],
                shared_max[tid + stride]
            );
        }
        __syncthreads();
    }

    float row_max = shared_max[0];

    // Compute exp(x - max) and the local sum.
    float local_sum = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        float value = expf(matrix[r * cols + c] - row_max);
        matrix[r * cols + c] = value;
        local_sum += value;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce the sum across the block.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared_sum[0];

    // Normalize the row.
    for (int c = tid; c < cols; c += blockDim.x) {
        matrix[r * cols + c] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;

        for (int j = 0; j < seq_len; ++j) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }

        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v,
                     float* d_out, int seq_len, int head_dim) {
    float* d_scores = nullptr;

    size_t scores_size =
        (size_t)seq_len * (size_t)seq_len * sizeof(float);

    cudaMalloc(&d_scores, scores_size);

    // Stage 1: QK^T / sqrt(head_dim)
    dim3 block_qk(16, 16);
    dim3 grid_qk(
        (seq_len + block_qk.x - 1) / block_qk.x,
        (seq_len + block_qk.y - 1) / block_qk.y
    );

    qk_scores<<<grid_qk, block_qk>>>(
        d_q, d_k, d_scores, seq_len, head_dim
    );

    // Stage 2: row-wise numerically stable softmax
    const int softmax_threads = 256;

    dim3 block_softmax(softmax_threads);
    dim3 grid_softmax(seq_len);

    softmax_rows<<<grid_softmax, block_softmax>>>(
        d_scores, seq_len, seq_len
    );

    // Stage 3: P * V
    dim3 block_pv(16, 16);
    dim3 grid_pv(
        (head_dim + block_pv.x - 1) / block_pv.x,
        (seq_len + block_pv.y - 1) / block_pv.y
    );

    pv_matmul<<<grid_pv, block_pv>>>(
        d_scores, d_v, d_out, seq_len, head_dim
    );

    // Free intermediate attention-score/probability matrix.
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int i = 0; i < head_dim; ++i) {
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int tile_size = tile_rows * tile_cols;

    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int tile_row = idx / tile_cols;
        int tile_col = idx % tile_cols;

        int src_row = src_row_start + tile_row;
        int src_col = src_col_start + tile_col;

        if (src_row < src_rows && src_col < src_cols) {
            shared_dst[idx] = src[src_row * src_cols + src_col];
        } else {
            shared_dst[idx] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int tile_size = tile_q * tile_k;

    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float dot = 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            dot += q_tile[i * head_dim + d] *
                   k_tile[j * head_dim + d];
        }

        s_tile[idx] = dot * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float max_val = s_tile[r * tile_k];

        for (int c = 1; c < tile_k; ++c) {
            max_val = fmaxf(max_val, s_tile[r * tile_k + c]);
        }

        row_max_out[r] = max_val;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int tile_size = tile_q * tile_k;

    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int r = idx / tile_k;

        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float sum = 0.0f;

        for (int c = 0; c < tile_k; ++c) {
            sum += p_tile[r * tile_k + c];
        }

        row_sum_out[r] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile,
                              float* out_acc, int tile_q, int tile_k,
                              int head_dim, int thread_id, int num_threads) {
    int total = tile_q * head_dim;

    for (int idx = thread_id; idx < total; idx += num_threads) {
        int q = idx / head_dim;
        int d = idx % head_dim;

        float sum = 0.0f;

        for (int k = 0; k < tile_k; ++k) {
            sum += p_tile[q * tile_k + k] *
                   v_tile[k * head_dim + d];
        }

        out_acc[q * head_dim + d] += sum;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    int block_row = blockIdx.x;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    int q_row_start = block_row * tile_q;

    /*
        Shared-memory layout:

        q_tile      : tile_q * head_dim
        k_tile      : tile_k * head_dim
        v_tile      : tile_k * head_dim
        s_tile      : tile_q * tile_k
        row_max     : tile_q
        row_sum     : tile_q
        running_max : tile_q
        running_sum : tile_q
        out_acc     : tile_q * head_dim
    */
    extern __shared__ float smem[];

    float* q_tile = smem;
    float* k_tile = q_tile + tile_q * head_dim;
    float* v_tile = k_tile + tile_k * head_dim;
    float* s_tile = v_tile + tile_k * head_dim;

    float* row_max = s_tile + tile_q * tile_k;
    float* row_sum = row_max + tile_q;

    float* running_max = row_sum + tile_q;
    float* running_sum = running_max + tile_q;

    float* out_acc = running_sum + tile_q;

    // ------------------------------------------------------------
    // Load Q tile once.
    // ------------------------------------------------------------
    load_tile(q, q_tile,
              q_row_start, 0,
              seq_len, head_dim,
              tile_q, head_dim,
              thread_id, num_threads);

    __syncthreads();

    // ------------------------------------------------------------
    // Initialize running softmax state and output accumulator.
    // ------------------------------------------------------------
    for (int r = thread_id; r < tile_q; r += num_threads) {
        running_max[r] = -3.402823466e+38F;
        running_sum[r] = 0.0f;
    }

    for (int idx = thread_id;
         idx < tile_q * head_dim;
         idx += num_threads) {
        out_acc[idx] = 0.0f;
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Stream over K/V tiles.
    // ------------------------------------------------------------
    for (int k_start = 0; k_start < seq_len; k_start += tile_k) {

        // Load K tile.
        load_tile(k, k_tile,
                  k_start, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  thread_id, num_threads);

        // Load V tile.
        load_tile(v, v_tile,
                  k_start, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  thread_id, num_threads);

        __syncthreads();

        // --------------------------------------------------------
        // Compute scaled QK^T for this tile.
        // --------------------------------------------------------
        tile_scores(q_tile, k_tile, s_tile,
                    tile_q, tile_k, head_dim, scale,
                    thread_id, num_threads);

        __syncthreads();

        // --------------------------------------------------------
        // Mask keys outside the valid sequence length.
        // --------------------------------------------------------
        for (int idx = thread_id;
             idx < tile_q * tile_k;
             idx += num_threads) {

            int k_col = k_start + (idx % tile_k);

            if (k_col >= seq_len) {
                s_tile[idx] = -3.402823466e+38F;
            }
        }

        __syncthreads();

        // --------------------------------------------------------
        // Find maximum score for each query row in this tile.
        // --------------------------------------------------------
        tile_rowmax(s_tile, row_max,
                    tile_q, tile_k,
                    thread_id, num_threads);

        __syncthreads();

        // --------------------------------------------------------
        // Compute exp(score - tile_max).
        // s_tile now contains the tile-local probabilities.
        // --------------------------------------------------------
        tile_exp(s_tile, row_max,
                 tile_q, tile_k,
                 thread_id, num_threads);

        __syncthreads();

        // --------------------------------------------------------
        // Compute sum(exp(score - tile_max)) for each row.
        // --------------------------------------------------------
        tile_rowsum(s_tile, row_sum,
                    tile_q, tile_k,
                    thread_id, num_threads);

        __syncthreads();

        // --------------------------------------------------------
        // Update running online-softmax state.
        //
        // Correct recurrence:
        //
        // new_max = max(old_max, tile_max)
        //
        // old correction:
        //     exp(old_max - new_max)
        //
        // tile correction:
        //     exp(tile_max - new_max)
        //
        // new_sum =
        //     old_sum * old_correction
        //     + tile_sum * tile_correction
        // --------------------------------------------------------
        for (int r = thread_id; r < tile_q; r += num_threads) {

            float old_max = running_max[r];
            float tile_max = row_max[r];

            float new_max = online_max(old_max, tile_max);

            float old_correction =
                correction_factor(old_max, new_max);

            float tile_correction =
                correction_factor(tile_max, new_max);

            running_sum[r] =
                update_running_sum(
                    running_sum[r],
                    old_correction,
                    row_sum[r] * tile_correction
                );

            // Rescale previously accumulated output into
            // the new maximum frame.
            rescale_output(
                out_acc + r * head_dim,
                head_dim,
                old_correction
            );

            running_max[r] = new_max;
        }

        __syncthreads();

        // --------------------------------------------------------
        // Rescale the CURRENT tile from tile_max to new_max.
        //
        // tile_exp produced:
        //
        //     exp(score - tile_max)
        //
        // but we need:
        //
        //     exp(score - new_max)
        //
        // Therefore multiply by:
        //
        //     exp(tile_max - new_max)
        // --------------------------------------------------------
        for (int idx = thread_id;
             idx < tile_q * tile_k;
             idx += num_threads) {

            int r = idx / tile_k;

            float tile_correction =
                correction_factor(row_max[r], running_max[r]);

            s_tile[idx] *= tile_correction;
        }

        __syncthreads();

        // --------------------------------------------------------
        // Accumulate the correctly rescaled P_tile * V_tile.
        // --------------------------------------------------------
        accumulate_pv(
            s_tile,
            v_tile,
            out_acc,
            tile_q,
            tile_k,
            head_dim,
            thread_id,
            num_threads
        );

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Normalize the final accumulated output.
    // ------------------------------------------------------------
    for (int idx = thread_id;
         idx < tile_q * head_dim;
         idx += num_threads) {

        int r = idx / head_dim;
        int d = idx % head_dim;

        int global_row = q_row_start + r;

        if (global_row < seq_len) {
            out[global_row * head_dim + d] =
                out_acc[idx] / running_sum[r];
        }
    }
}

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

