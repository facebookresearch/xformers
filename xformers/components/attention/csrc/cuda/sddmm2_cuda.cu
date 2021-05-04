// taken from
// https://github.com/hgyhungry/ge-spmm/blob/master/pytorch-custom/sddmm.cu with
// slight modifications to add batch support
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include "computeUtil.h"

namespace ge_spmm {

template <typename T>
__global__ void sddmmCSR2Scale(
    const int S_mrows,
    int D_kcols,
    int S_nrows,
    const unsigned long Size,
    int* S_csrRowPtr,
    int* S_csrColInd,
    T* D1_dnVal_,
    T* D2_dnVal_,
    T* O_csrVal_) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x << 1;

  int batch_id = blockIdx.y;
  int batch_offset_1 = batch_id * S_mrows * D_kcols;
  int batch_offset_2 = batch_id * S_nrows * D_kcols;
  int batch_offset_out = batch_id * Size;

  T* D1_dnVal = D1_dnVal_ + batch_offset_1;
  T* D2_dnVal = D2_dnVal_ + batch_offset_2;
  T* O_csrVal = O_csrVal_ + batch_offset_out;

  if (blockIdx.x < Size / 16) {
    T multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float2 D1tmp[4], D2tmp[4];
    Load<int4, int>(offset2, S_csrColInd, eid);
    offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
    offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
    offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);
    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float2, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float2, float>(D2tmp, D2_dnVal, offset2, cid);
      vec2Dot4<float2>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = D_kcols & 31;
    if (res) {
      int cid2 = threadIdx.x + D_kcols - res;
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      for (int i = 0; i < (res >> 4) + 1; i++) {
        if ((i << 4) + threadIdx.x < res) {
          Load4<float, float>(D1, D1_dnVal, offset1, cid2);
          Load4<float, float>(D2, D2_dnVal, offset2, cid2);
          Dot4<float>(multi, D1, D2);
          cid2 += 16;
        }
      }
    }
    AllReduce4<float>(multi, 8, 32);
    if (threadIdx.x == 0) {
      Store<float4, float>(O_csrVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
    int offset2 = S_csrColInd[eid] * D_kcols;
    T multi = 0;
    int off1 = cid = (threadIdx.y << 4) + threadIdx.x;
    float D1tmp0, D2tmp0;
    for (int cc = 0; cc < (D_kcols >> 5); cc++) {
      D1tmp0 = D1_dnVal[offset1 + cid];
      D2tmp0 = D2_dnVal[offset2 + cid];
      multi += D1tmp0 * D2tmp0;
      cid += 32;
    }
    int res = D_kcols & 31;
    D1tmp0 = D2tmp0 = 0;
    if (res) {
      if (off1 < res) {
        D1tmp0 = D1_dnVal[offset1 + cid];
        D2tmp0 = D2_dnVal[offset2 + cid];
      }
      multi += D1tmp0 * D2tmp0;
    }
    for (int stride = 16; stride > 0; stride >>= 1) {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      O_csrVal[eid] = multi;
    }
  }
}

template <typename T>
__global__ void sddmmCSR1Scale(
    const int S_mrows,
    int D_kcols,
    int S_nrows,
    const unsigned long Size,
    int* S_csrRowPtr,
    int* S_csrColInd,
    T* D1_dnVal_,
    T* D2_dnVal_,
    T* O_csrVal_) {
  int eid = (blockIdx.x << 4) + (threadIdx.y << 2);
  int cid = threadIdx.x;

  int batch_id = blockIdx.y;
  int batch_offset_1 = batch_id * S_mrows * D_kcols;
  int batch_offset_2 = batch_id * S_nrows * D_kcols;
  int batch_offset_out = batch_id * Size;

  T* D1_dnVal = D1_dnVal_ + batch_offset_1;
  T* D2_dnVal = D2_dnVal_ + batch_offset_2;
  T* O_csrVal = O_csrVal_ + batch_offset_out;

  if (blockIdx.x < Size / 16) {
    T multi[4] = {0, 0, 0, 0};
    int offset1[4], offset2[4];
    float D1tmp[4], D2tmp[4];

    Load<int4, int>(offset2, S_csrColInd, eid);

    offset1[0] = findRow(S_csrRowPtr, eid, 0, S_mrows);
    offset1[3] = findRow(S_csrRowPtr, eid + 3, offset1[0], S_mrows);
    offset1[1] = findRow(S_csrRowPtr, eid + 1, offset1[0], offset1[3]);
    offset1[2] = findRow(S_csrRowPtr, eid + 2, offset1[1], offset1[3]);

    selfMulConst4<int>(offset1, D_kcols);
    selfMulConst4<int>(offset2, D_kcols);

    for (int i = 0; i < (D_kcols >> 5); i++) {
      Load4<float, float>(D1tmp, D1_dnVal, offset1, cid);
      Load4<float, float>(D2tmp, D2_dnVal, offset2, cid);
      Dot4<float>(multi, D1tmp, D2tmp);
      cid += 32;
    }
    int res = D_kcols & 31;
    if (res) {
      float D1[4] = {0, 0, 0, 0}, D2[4] = {0, 0, 0, 0};
      if (threadIdx.x < res) {
        Load4<float, float>(D1, D1_dnVal, offset1, cid);
        Load4<float, float>(D2, D2_dnVal, offset2, cid);
        Dot4<float>(multi, D1, D2);
      }
    }
    AllReduce4<float>(multi, 16, 32);
    if (threadIdx.x == 0) {
      Store<float4, float>(O_csrVal, multi, eid);
    }
  } else // Dynamic parrallel?
  {
    eid = Size - (Size & 15) + (blockIdx.x - (Size / 16));
    int offset1 = findRow(S_csrRowPtr, eid, 0, S_mrows) * D_kcols;
    int offset2 = S_csrColInd[eid] * D_kcols;
    T multi = 0;
    int off1 = cid = threadIdx.x;
    float D1tmp0, D2tmp0;
    for (int cc = 0; cc < (D_kcols >> 5); cc++) {
      D1tmp0 = D1_dnVal[offset1 + cid];
      D2tmp0 = D2_dnVal[offset2 + cid];
      multi += D1tmp0 * D2tmp0;
      cid += 32;
    }
    int res = D_kcols & 31;
    D1tmp0 = D2tmp0 = 0;
    if (res) {
      if (off1 < res) {
        D1tmp0 = D1_dnVal[offset1 + cid];
        D2tmp0 = D2_dnVal[offset2 + cid];
      }
      multi += D1tmp0 * D2tmp0;
    }
    for (int stride = 16; stride > 0; stride >>= 1) {
      multi += __shfl_xor_sync(0xffffffff, multi, stride, 32);
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      O_csrVal[eid] = multi;
    }
  }
}

torch::Tensor sddmm_cuda_csr(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor D1,
    torch::Tensor D2) {
  const auto batch_size = D1.size(0);
  const auto m = D1.size(1);
  const auto k = D1.size(2);
  const auto n = D2.size(1);
  const auto nnz = colind.size(0);
  auto out = torch::empty({batch_size, nnz}, D1.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid_dim(nnz / 16 + (nnz & 15), batch_size, 1);

  if ((k % 2) == 0) {
    dim3 block_dim(16, 4, 1);
    sddmmCSR2Scale<<<grid_dim, block_dim, 0, stream>>>(
        m,
        k,
        n,
        nnz,
        rowptr.data_ptr<int>(),
        colind.data_ptr<int>(),
        D1.data_ptr<float>(),
        D2.data_ptr<float>(),
        out.data_ptr<float>());
  } else {
    dim3 block_dim(32, 4, 1);
    sddmmCSR1Scale<<<grid_dim, block_dim, 0, stream>>>(
        m,
        k,
        n,
        nnz,
        rowptr.data_ptr<int>(),
        colind.data_ptr<int>(),
        D1.data_ptr<float>(),
        D2.data_ptr<float>(),
        out.data_ptr<float>());
  }
  return out;
}
} // namespace ge_spmm

torch::Tensor csr_sddmm(
    torch::Tensor D1,
    torch::Tensor D2,
    torch::Tensor unused,
    torch::Tensor rowptr,
    torch::Tensor colind) {
  TORCH_CHECK(rowptr.device().type() == torch::kCUDA);
  TORCH_CHECK(colind.device().type() == torch::kCUDA);
  TORCH_CHECK(D1.device().type() == torch::kCUDA);
  TORCH_CHECK(D2.device().type() == torch::kCUDA);
  TORCH_CHECK(rowptr.is_contiguous());
  TORCH_CHECK(colind.is_contiguous());
  TORCH_CHECK(D1.is_contiguous());
  TORCH_CHECK(D2.is_contiguous());
  TORCH_CHECK(rowptr.dtype() == torch::kInt32);
  TORCH_CHECK(colind.dtype() == torch::kInt32);
  TORCH_CHECK(D1.dtype() == torch::kFloat32);
  TORCH_CHECK(D2.dtype() == torch::kFloat32);
  return ge_spmm::sddmm_cuda_csr(rowptr, colind, D1, D2);
}

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::csr_sddmm(Tensor a, Tensor b, Tensor row_indices, Tensor row_offsets, Tensor column_indices) -> Tensor"));
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("xformers::csr_sddmm"), TORCH_FN(csr_sddmm));
}
