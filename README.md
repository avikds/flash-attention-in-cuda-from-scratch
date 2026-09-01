# Flash Attention in CUDA from Scratch

Build a tiled, IO-aware Flash Attention implementation in CUDA, starting from elementary GPU primitives and progressing to a fused online-softmax attention kernel. Along the way you implement a naive attention baseline, the online softmax math, and finish with a causal variant suitable for autoregressive models.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** vector_add
- [x] **2.** scale_array
- [x] **3.** elementwise_exp
- [x] **4.** row_max
- [x] **5.** row_sum
- [x] **6.** dot_product
- [x] **7.** matmul
- [x] **8.** transpose
- [x] **9.** qk_scores
- [x] **10.** softmax_rows
- [x] **11.** pv_matmul
- [x] **12.** naive_attention
- [x] **13.** online_max
- [x] **14.** correction_factor
- [x] **15.** update_running_sum
- [x] **16.** rescale_output
- [x] **17.** load_tile
- [x] **18.** tile_scores
- [x] **19.** tile_rowmax
- [x] **20.** tile_exp
- [x] **21.** tile_rowsum
- [x] **22.** accumulate_pv
- [x] **23.** flash_attention_kernel
- [x] **24.** flash_attention_launcher
- [x] **25.** causal_mask
- [x] **26.** flash_attention_causal_kernel

## Results

```
vector_add+scale_array[0..3]: 8.00 8.00 8.00 8.00
Q row0 max=0.3402 sum=0.5278

--- Attention outputs (seq_len=8, head_dim=4) ---
naive  row 0:  0.0950  0.0663 -0.1126  0.0952 
flash  row 0:  0.0950  0.0663 -0.1126  0.0952 
causal row 0:  0.2831 -0.3024 -0.2222  0.1289 
naive  row 7:  0.0810  0.0570 -0.1095  0.0987 
flash  row 7:  0.0810  0.0570 -0.1095  0.0987 
causal row 7:  0.0810  0.0570 -0.1095  0.0987 

max|naive - flash| = 2.235174e-08

--- Memory: naive O(N^2) scores vs flash O(1) global scratch ---
  this run (seq_len=8): naive scores = 256 bytes, flash global scratch = 0
       seq_len       naive scores      flash scratch
          1024             4.2 MB    ~0 (tiles only)
          8192           268.4 MB    ~0 (tiles only)
         32768          4295.0 MB    ~0 (tiles only)
        131072         68719.5 MB    ~0 (tiles only)
  Flash keeps only a tile in shared memory (tens of KB per block), so it runs
  at sequence lengths where the naive score matrix would not fit in GPU memory.
  (This from-scratch kernel favors clarity over speed; it is not throughput-
   optimized like production FlashAttention -- the win here is memory scaling.)
```
