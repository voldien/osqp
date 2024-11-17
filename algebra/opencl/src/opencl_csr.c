/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "opencl_csr.h"
#include "opencl_alloc.h"
#include "opencl_helper.h"
#include "opencl_lin_alg.h"

// #include "csr_type.h"
#include "glob_opts.h"
#include "osqp_api_types.h"
#include <CL/cl.h>

extern OpenCL_Handle_t *handle;

/* This function is implemented in cl_lin_alg.cu */
extern void scatter(cl_mem out, const cl_mem in, const cl_mem ind, OSQPInt n);

/*******************************************************************************
 *                         Private Functions                                   *
 *******************************************************************************/

static void init_SpMV_interface(csr *M) {

  OSQPFloat *d_x;
  OSQPFloat *d_y;
  cl_mem vecx, vecy;
  //
  OSQPFloat alpha = 1.0;
  const OSQPInt m = M->m;
  const OSQPInt n = M->n;

  /* Only create the matrix if it has non-zero dimensions.
   * Some versions of CUDA don't allow creating matrices with rows/columns of
   * size 0 and assert instead. So we don't create the matrix object, and
   * instead will never perform any operations on it.
   */

  // TODO: add me
  if ((m > 0) && (n > 0)) {
    //  /* Wrap raw data into cuSPARSE API matrix */
    //  checkCudaErrors(cusparseCreateCsr(
    //    &M->SpMatDescr, m, n, M->nnz,
    //    (void*)M->row_ptr, (void*)M->col_ind, (void*)M->val,
    //    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //    CUSPARSE_INDEX_BASE_ZERO, CUDA_FLOAT));
    //

    if (!M->SpMatBufferSize) {
      //    cl_malloc((void **) &d_x, n * sizeof(OSQPFloat));
      //    cl_malloc((void **) &d_y, m * sizeof(OSQPFloat));
      //
      cl_vecf_create(&vecx, d_x, n);
      cl_vecf_create(&vecy, d_y, m);
      // cl_mat_Axpy(const csr *A, const OSQPVectorf *vecx, OSQPVectorf *vecy,
      // OSQPFloat alpha, OSQPFloat beta)

      //
      //     /* Allocate workspace for cusparseSpMV */
      //     checkCudaErrors(cusparseSpMV_bufferSize(
      //       CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      //       &alpha, M->SpMatDescr, vecx, &alpha, vecy,
      //       CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, &M->SpMatBufferSize));
      //
      //     if (M->SpMatBufferSize){
      //       cl_malloc((void **) &M->SpMatBuffer, M->SpMatBufferSize);
      // }
      //
      //     cl_vec_destroy(vecx);
      //     cl_vec_destroy(vecy);
      //
      //     cl_free((void **) &d_x);
      //     cl_free((void **) &d_y);
    }
  }
}

/*
 *  Creates a CSR matrix with the specified dimension (m,n,nnz).
 *
 *  If specified, it allocates proper amount of device memory
 *  allocate_on_device = 1: device memory for CSR
 *  allocate_on_device = 2: device memory for CSR (+ col_ind)
 */
csr *csr_alloc(OSQPInt m, OSQPInt n, OSQPInt nnz, OSQPInt allocate_on_device) {

  csr *dev_mat = (csr *)c_calloc(1, sizeof(csr));

  if (!dev_mat) {
    return NULL;
  }

  dev_mat->m = m;
  dev_mat->n = n;
  dev_mat->nnz = nnz;

  if (allocate_on_device > 0) {

    const unsigned int valSize =
        align((dev_mat->nnz + 1) * sizeof(OSQPFloat),
              handle->deviceInformations[0].alignedSize);
    const unsigned int rowSize =
        align((dev_mat->m + 1) * sizeof(OSQPInt),
              handle->deviceInformations[0].alignedSize);
    const unsigned int colindSize =
        align(dev_mat->nnz * sizeof(OSQPInt),
              handle->deviceInformations[0].alignedSize);
    const unsigned int row_ind_size =
        align(dev_mat->nnz * sizeof(OSQPInt),
              handle->deviceInformations[0].alignedSize);

    const size_t full_size =
        align(valSize + rowSize + colindSize + row_ind_size,
              handle->deviceInformations[0].alignedSize);

    dev_mat->mainbuffer = cl_allocate_mem(
        handle->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, full_size);

    size_t nrRegions = 3;
    const cl_buffer_region region[4] = {
        {0, valSize},
        /*  */
        {valSize, rowSize},

        {valSize + rowSize, colindSize},
        {valSize + rowSize + colindSize, row_ind_size}};

    if (allocate_on_device > 1) {
      nrRegions += 1;
    }

    cl_allocate_sub_mem(dev_mat->mainbuffer,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, nrRegions,
                        region, &dev_mat->cl_val);

    /*  */
    dev_mat->val = cl_host_map(dev_mat->mainbuffer, CL_MAP_READ, &region[0]);
    dev_mat->row_ptr =
        cl_host_map(dev_mat->mainbuffer, CL_MAP_READ, &region[1]);
    dev_mat->col_ind =
        cl_host_map(dev_mat->mainbuffer, CL_MAP_READ, &region[2]);

    /*  Check if success.*/
    if (allocate_on_device > 1) {
      dev_mat->row_ind =
          cl_host_map(dev_mat->mainbuffer, CL_MAP_READ, &region[3]);
    }

  } else {

    dev_mat->mainbuffer = NULL;
    dev_mat->cl_val = NULL;
    dev_mat->cl_row_ptr = NULL;
    dev_mat->cl_col_ind = NULL;
    dev_mat->cl_row_ind = NULL;

    dev_mat->val = NULL;
    dev_mat->row_ptr = NULL;
    dev_mat->col_ind = NULL;
    dev_mat->row_ind = NULL;
  }

  dev_mat->SpMatBufferSize = 0;

  return dev_mat;
}

csr *csr_init(OSQPInt m, OSQPInt n, const OSQPInt *h_row_ptr,
              const OSQPInt *h_col_ind, const OSQPFloat *h_val) {

  csr *dev_mat = csr_alloc(m, n, h_row_ptr[m], 1);

  if (!dev_mat) {
    return NULL;
  }

  if (m == 0) {
    return dev_mat;
  }

  /* copy_matrix_to_device */
  if (h_row_ptr) {
    cl_memcpy_h2d(dev_mat->cl_row_ptr, h_row_ptr,
                  (dev_mat->m + 1) * sizeof(OSQPInt));
  }
  if (h_col_ind) {
    cl_memcpy_h2d(dev_mat->cl_col_ind, h_col_ind,
                  dev_mat->nnz * sizeof(OSQPInt));
  }
  if (h_val) {
    cl_memcpy_h2d(dev_mat->cl_val, h_val, dev_mat->nnz * sizeof(OSQPFloat));
  }

  return dev_mat;
}

static void coo2csr(const cl_mem csr_row, OSQPInt nnz, OSQPInt num_rows_m,
                    cl_mem cooRowInd) {
  // offset = 2 * nnz_triu;
  int number_of_blocks =
      (num_rows_m / handle->deviceInformations[0].localworksize[0]) + 1;

  const cl_kernel kernel = handle->offsets_to_indices_kernel;

  cl_int err = clSetKernelArg(kernel, 0, sizeof(num_rows_m), &num_rows_m);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &csr_row); /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cooRowInd);

  size_t workgroup[1] = {number_of_blocks};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                               &local[0], 0, 0, 0);
}

/*
 *  Compress row indices from the COO format to the row pointer
 *  of the CSR format.
 */
void compress_row_ind(csr *mat) {

  // TODO: add me
  // cl_free((void** ) &mat->row_ptr);
  // cl_malloc((void** ) &mat->row_ptr, (mat->m + 1) * sizeof(OSQPFloat));

  coo2csr(mat->cl_row_ind, mat->nnz, mat->m, mat->cl_row_ptr);
}

static void csr2coo(const cl_mem csr_row, OSQPInt nnz, OSQPInt num_rows_m,
                    cl_mem cooRowInd) {
  // offset = 2 * nnz_triu;
  int number_of_blocks =
      (num_rows_m / handle->deviceInformations[0].localworksize[0]) + 1;

  const cl_kernel kernel = handle->offsets_to_indices_kernel;

  cl_int err = clSetKernelArg(kernel, 0, sizeof(num_rows_m), &num_rows_m);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &csr_row); /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cooRowInd);

  size_t workgroup[1] = {number_of_blocks};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                               &local[0], 0, 0, 0);
}

void csr_expand_row_ind(csr *mat) {

  if (!mat->row_ind && !mat->cl_row_ind) {

    const unsigned int valSize =
        align((mat->nnz + 1) * sizeof(OSQPFloat),
              handle->deviceInformations[0].alignedSize);
    const unsigned int rowSize =
        align((mat->m + 1) * sizeof(OSQPInt),
              handle->deviceInformations[0].alignedSize);
    const unsigned int colindSize = align(
        mat->nnz * sizeof(OSQPInt), handle->deviceInformations[0].alignedSize);
    const unsigned int row_ind_size = align(
        mat->nnz * sizeof(OSQPInt), handle->deviceInformations[0].alignedSize);

    const cl_buffer_region region =
        // TODO: fix regions
        {valSize + rowSize + colindSize, row_ind_size};
    cl_allocate_sub_mem(mat->mainbuffer, CL_MEM_READ_WRITE, 1, &region,
                        &mat->cl_row_ind);
    mat->row_ind = cl_host_map(mat->mainbuffer, CL_MAP_READ, &region);

    /*  CSR to COO. */
    csr2coo(mat->cl_row_ptr, mat->nnz, mat->m, mat->cl_row_ind);
  }
}

/*
 *  Sorts matrix in COO format by row. It returns a permutation
 *  vector that describes reordering of the elements.
 */
cl_mem coo_sort(csr *A) {

  cl_mem A_to_At_permutation;
  cl_mem pBuffer;
  size_t pBufferSizeInBytes = 1;

  A_to_At_permutation = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                        A->nnz * sizeof(OSQPInt));

  // checkCudaErrors(cusparseCreateIdentityPermutation(CUDA_handle->cusparseHandle,
  // A->nnz, A_to_At_permutation));
  //

  // checkCudaErrors(cusparseXcoosort_bufferSizeExt(CUDA_handle->cusparseHandle,
  // A->m, A->n, A->nnz, A->row_ind, A->col_ind, &pBufferSizeInBytes));
  //
  pBuffer = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                            pBufferSizeInBytes * sizeof(char));
  //
  // checkCudaErrors(cusparseXcoosortByRow(CUDA_handle->cusparseHandle, A->m,
  // A->n, A->nnz, A->row_ind, A->col_ind, A_to_At_permutation, pBuffer));
  //
  cl_free(pBuffer);

  return A_to_At_permutation;
}

/*
 * Compute transpose of a matrix in COO format.
 */
void coo_tranpose(csr *A) {
  OSQPInt m = A->m;
  A->m = A->n;
  A->n = m;

  /*  Swap pointer. */
  OSQPInt *row_ind = A->row_ind;
  A->row_ind = A->col_ind;
  A->col_ind = row_ind;

  /*  Swap memory. */
  cl_mem row_ind_cl = A->cl_row_ind;
  A->cl_row_ind = A->cl_col_ind;
  A->cl_col_ind = row_ind_cl;
}

/*
 *  values[i] = values[permutation[i]] for i in [0,n-1]
 */
void permute_vector(cl_mem values, const cl_mem permutation, OSQPInt n) {

  cl_mem permuted_values;
  permuted_values = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                    n * sizeof(OSQPFloat));

  cl_vec_gather(n, values, permuted_values, permutation);

  cl_memcpy_d2d(values, permuted_values, n * sizeof(OSQPFloat));
  cl_free(permuted_values);
}

/*
 *  Copy the values and pointers form target to the source matrix.
 *  The device memory of source has to be freed first to avoid a
 *  memory leak in case it holds allocated memory.
 *
 *  The MatrixDescription has to be destroyed first since it is a
 *  pointer hidded by a typedef.
 *
 *  The pointers of source matrix are set to NULL to avoid
 *  accidental freeing of the associated memory blocks.
 */
void copy_csr(csr *target, csr *source) {

  target->m = source->m;
  target->n = source->n;
  target->nnz = source->nnz;

  // cl_free(target->cl_val);
  // cl_free(target->cl_row_ind);
  // cl_free(target->cl_row_ptr);
  // cl_free(target->cl_col_ind);
  // cl_free(target->mainbuffer);

  target->val = source->val;
  target->row_ind = source->row_ind;
  target->row_ptr = source->row_ptr;
  target->col_ind = source->col_ind;

  target->cl_val = source->cl_val;
  target->cl_row_ind = source->cl_row_ind;
  target->cl_row_ptr = source->cl_row_ptr;
  target->cl_col_ind = source->cl_col_ind;
  target->mainbuffer = source->mainbuffer;

  source->val = NULL;
  source->row_ind = NULL;
  source->row_ptr = NULL;
  source->col_ind = NULL;

  target->cl_val = NULL;
  target->cl_row_ind = NULL;
  target->cl_row_ptr = NULL;
  target->cl_col_ind = NULL;
  target->mainbuffer = NULL;
}

void csr_triu_to_full(csr *P_triu, cl_mem *P_triu_to_full_permutation,
                      cl_mem *P_diag_indices) {

  OSQPInt number_of_blocks;
  cl_mem has_non_zero_diag_element;
  cl_mem d_nnz_diag;
  cl_mem d_P;
  OSQPInt h_nnz_diag, Full_nnz, nnz_triu, n, nnz_max_Full;
  OSQPInt offset;

  nnz_triu = P_triu->nnz;
  n = P_triu->n;
  nnz_max_Full = 2 * nnz_triu + n;

  csr *Full_P = csr_alloc(n, n, nnz_max_Full, 2);

  has_non_zero_diag_element = cl_allocate_calloc_mem(
      handle->context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
      n * sizeof(OSQPInt));
  d_nnz_diag = cl_allocate_calloc_mem(handle->context,
                                      CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                                      sizeof(OSQPInt));

  csr_expand_row_ind(P_triu);

  {
    const cl_kernel kernel = handle->fill_full_matrix_kernel;
    /*  */
    cl_int err =
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &Full_P->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &Full_P->cl_col_ind);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_nnz_diag);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &has_non_zero_diag_element);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &P_triu->cl_row_ind);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &P_triu->cl_col_ind);
    err = clSetKernelArg(kernel, 6, sizeof(nnz_triu), &nnz_triu);
    err = clSetKernelArg(kernel, 7, sizeof(n), &n);

    number_of_blocks =
        (nnz_triu / handle->deviceInformations[0].localworksize[0]) + 1;
    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }

  {
    offset = 2 * nnz_triu;
    number_of_blocks = (n / handle->deviceInformations[0].localworksize[0]) + 1;

    const cl_kernel kernel = handle->add_diagonal_kernel;

    cl_int err =
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &Full_P->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &Full_P->cl_col_ind);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &has_non_zero_diag_element);
    err = clSetKernelArg(kernel, 3, sizeof(n), &n);
    err = clSetKernelArg(kernel, 4, sizeof(offset), &offset);

    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }
  /* The Full matrix now is of size (2n)x(2n)
   *                  [P 0]
   *                  [0 D]
   * where P is the desired full matrix and D is
   * a diagonal that contains dummy values
   */
  cl_memcpy_d2h(&h_nnz_diag, d_nnz_diag, sizeof(OSQPInt));

  {
    Full_nnz = (2 * (nnz_triu - h_nnz_diag)) + n;
    d_P = coo_sort(Full_P);

    const cl_kernel kernel = handle->reduce_permutation_kernel;

    cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_P); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &nnz_triu);
    err = clSetKernelArg(kernel, 2, sizeof(Full_nnz), &Full_nnz);

    number_of_blocks =
        (nnz_triu / handle->deviceInformations[0].localworksize[0]) + 1;

    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);

    /* permute vector */
    cl_vec_gather(Full_nnz, P_triu->cl_val, Full_P->cl_val, d_P);

    *P_triu_to_full_permutation = cl_allocate_mem(
        handle->context, CL_MEM_READ_WRITE, Full_nnz * sizeof(OSQPInt));

    cl_memcpy_d2h(*P_triu_to_full_permutation, d_P, Full_nnz * sizeof(OSQPInt));
  }

  {
    *P_diag_indices = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                      n * sizeof(OSQPInt));

    number_of_blocks =
        (Full_nnz / handle->deviceInformations[0].localworksize[0]) + 1;

    const cl_kernel kernel = handle->get_diagonal_indices_kernel;

    cl_int err =
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &Full_P->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &Full_P->cl_col_ind);
    err = clSetKernelArg(kernel, 2, sizeof(Full_nnz), &Full_nnz);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &(*P_diag_indices));

    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }

  Full_P->nnz = Full_nnz;
  compress_row_ind(Full_P);
  copy_csr(P_triu, Full_P);

  cl_mat_free(Full_P);
  cl_free(d_P);
  cl_free(d_nnz_diag);
  cl_free(has_non_zero_diag_element);
}

/**
 * Matrix A is converted from CSC to CSR. The data in A is interpreted as
 * being in CSC format, even if it is in CSR.
 * This operation is equivalent to a transpose. We temporarily allocate space
 * for the new matrix since this operation cannot be done inplace.
 * Additionally, a gather indices vector is generated to perform the conversion
 * from A to A' faster during a matrix update.
 */
void csr_transpose(csr *A, cl_mem *A_to_At_permutation) {

  (*A_to_At_permutation) = NULL;

  if (A->nnz == 0) {
    OSQPInt tmp = A->n;
    A->n = A->m;
    A->m = tmp;
    return;
  }

  csr_expand_row_ind(A);
  coo_tranpose(A);
  (*A_to_At_permutation) = coo_sort(A);
  compress_row_ind(A);

  permute_vector(A->cl_val, *A_to_At_permutation, A->nnz);
}

/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cl_mat_init_P(const OSQPCscMatrix *mat, csr **P, cl_mem *d_P_triu_val,
                   cl_mem *d_P_triu_to_full_ind, cl_mem *d_P_diag_ind) {

  OSQPInt n = mat->n;
  OSQPInt nnz = mat->p[n];

  /* Initialize upper triangular part of P */
  *P = csr_init(n, n, mat->p, mat->i, mat->x);

  /* Convert P to a full matrix. Store indices of diagonal and triu elements. */
  csr_triu_to_full(*P, d_P_triu_to_full_ind, d_P_diag_ind);
  csr_expand_row_ind(*P);

  /* We need 0.0 at val[nzz] -> nnz+1 elements */
  *d_P_triu_val = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                  (nnz + 1) * sizeof(OSQPFloat));
  cl_memcpy_h2d(*d_P_triu_val, mat->x, nnz * sizeof(OSQPFloat));

  init_SpMV_interface(*P);
}

void cl_mat_init_A(const OSQPCscMatrix *mat, csr **A, csr **At,
                   cl_mem *d_A_to_At_ind) {

  const OSQPInt m = mat->m;
  const OSQPInt n = mat->n;

  /* Initializing At is easy since it is equal to A in CSC */
  *At = csr_init(n, m, mat->p, mat->i, mat->x);
  csr_expand_row_ind(*At);

  /* We need to take transpose of At to get A */
  *A = csr_init(n, m, mat->p, mat->i, mat->x);
  csr_transpose(*A, d_A_to_At_ind);
  csr_expand_row_ind(*A);

  init_SpMV_interface(*A);
  init_SpMV_interface(*At);
}

void cl_mat_update_P(const OSQPFloat *Px, const OSQPInt *Px_idx, OSQPInt Px_n,
                     csr **P, OSQPFloat *d_P_triu_val,
                     OSQPInt *d_P_triu_to_full_ind, OSQPInt *d_P_diag_ind,
                     OSQPInt P_triu_nnz) {

  if (!Px_idx) { /* Update whole P */
    cl_mem d_P_val_new;

    /* Allocate memory */
    d_P_val_new = cl_allocate_mem(handle->context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                                  (P_triu_nnz + 1) * sizeof(OSQPFloat));

    /* Copy new values from host to device */
    cl_memcpy_h2d(d_P_val_new, Px, P_triu_nnz * sizeof(OSQPFloat));

    // FIX: cl_memcpy_h2d((*At)->cl_val, Ax, Annz * sizeof(OSQPFloat));

    cl_vec_gather((*P)->nnz, d_P_val_new, (*P)->cl_val, d_P_triu_to_full_ind);

    cl_free(d_P_val_new);

  } else { /* Update P partially */

    cl_mem d_P_val_new;
    cl_mem d_P_ind_new;

    /* Allocate memory */
    d_P_val_new = cl_allocate_mem(handle->context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                                  Px_n * sizeof(OSQPFloat));
    d_P_ind_new = cl_allocate_mem(handle->context,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                                  Px_n * sizeof(OSQPInt));

    /* Copy new values and indices from host to device */
    cl_memcpy_h2d(d_P_val_new, Px, Px_n * sizeof(OSQPFloat));
    cl_memcpy_h2d(d_P_ind_new, Px_idx, Px_n * sizeof(OSQPInt));

    /* Update d_P_triu_val */
    scatter(d_P_triu_val, d_P_val_new, d_P_ind_new, Px_n);

    /* Gather from d_P_triu_val to update full P */
    cl_vec_gather((*P)->nnz, d_P_triu_val, (*P)->cl_val, d_P_triu_to_full_ind);

    cl_free(d_P_val_new);
    cl_free(d_P_ind_new);
  }
}

void cl_mat_update_A(const OSQPFloat *Ax, const OSQPInt *Ax_idx, OSQPInt Ax_n,
                     csr **A, csr **At, cl_mem d_A_to_At_ind) {

  OSQPInt Annz = (*A)->nnz;
  cl_mem Aval = (*A)->cl_val;
  cl_mem Atval = (*At)->cl_val;

  if (!Ax_idx) { /* Update whole A */
    /* Updating At is easy since it is equal to A in CSC */
    cl_memcpy_h2d((*At)->cl_val, Ax, Annz * sizeof(OSQPFloat));

    /* Updating A requires transpose of A_new */
    cl_vec_gather(Annz, Atval, Aval, d_A_to_At_ind);
  } else { /* Update A partially */
    cl_mem d_At_val_new;
    cl_mem d_At_ind_new;

    /* Allocate memory */
    d_At_val_new = cl_allocate_mem(handle->context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                                   Ax_n * sizeof(OSQPFloat));
    d_At_ind_new = cl_allocate_mem(handle->context,
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                                   Ax_n * sizeof(OSQPInt));

    /* Copy new values and indices from host to device */
    cl_memcpy_h2d(d_At_val_new, Ax, Ax_n * sizeof(OSQPFloat));
    cl_memcpy_h2d(d_At_ind_new, Ax_idx, Ax_n * sizeof(OSQPInt));

    /* Update At first since it is equal to A in CSC */
    scatter(Atval, d_At_val_new, d_At_ind_new, Ax_n);

    cl_free(d_At_val_new);
    cl_free(d_At_ind_new);

    /* Gather from Atval to construct Aval */
    cl_vec_gather(Annz, Atval, Aval, d_A_to_At_ind);
  }
}

void cl_mat_free(csr *mat) {
  if (mat) {
    if (mat->val && mat->cl_val) {
      cl_host_ummap(mat->mainbuffer, mat->val);
      cl_free(mat->cl_val);
    }
    if (mat->row_ptr && mat->cl_row_ptr) {
      cl_host_ummap(mat->mainbuffer, mat->row_ptr);

      cl_free(mat->cl_row_ptr);
    }
    if (mat->col_ind && mat->cl_col_ind) {
      cl_host_ummap(mat->mainbuffer, mat->col_ind);
      cl_free(mat->cl_col_ind);
    }
    if (mat->row_ind && mat->cl_row_ind) {
      cl_host_ummap(mat->mainbuffer, mat->row_ind);
      cl_free(mat->cl_row_ind);
    }
    if (mat->mainbuffer) {
      cl_free(mat->mainbuffer);
    }

    c_free(mat);
  }
}

OSQPInt cl_csr_is_eq(const csr *A, const csr *B, OSQPFloat tol) {

  OSQPInt h_res = 0;

  // If number of columns, rows and non-zeros are not the same, they are not
  // equal.
  if ((A->n != B->n) || (A->m != B->m) || (A->nnz != B->nnz)) {
    return 0;
  }

  OSQPInt nnz = A->nnz;

  cl_int err;
  const cl_kernel kernel = handle->csr_eq_kernel;

  cl_mem tmp_result =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                      sizeof(OSQPInt));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->cl_row_ptr);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &A->cl_col_ind);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &A->cl_val);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &B->cl_row_ptr);
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &B->cl_col_ind);
  err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &B->cl_val);
  err = clSetKernelArg(kernel, 6, sizeof(A->m), &A->m);
  err = clSetKernelArg(kernel, 7, sizeof(tol), &tol);
  err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &tmp_result);

  size_t workgroup[1] = {
      (nnz / handle->deviceInformations[0].localworksize[0]) + 1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                               &local[0], 0, 0, 0);

  cl_memcpy_d2h(&h_res, tmp_result, sizeof(h_res));
  cl_free(tmp_result);
  h_res = (unsigned int)h_res;

  return h_res;
}

void cl_submat_byrows(const csr *A, const cl_mem d_rows, csr **Ared,
                      csr **Aredt) {

  OSQPInt new_m = 0;

  const OSQPInt n = A->n;
  const OSQPInt m = A->m;
  const OSQPInt nnz = A->nnz;

  cl_mem d_predicate;
  cl_mem d_compact_address;
  cl_mem d_row_predicate;
  cl_mem d_new_row_number;

  d_row_predicate =
      cl_allocate_mem(handle->context, CL_MEM_READ_WRITE, m * sizeof(OSQPInt));
  d_new_row_number =
      cl_allocate_mem(handle->context, CL_MEM_READ_WRITE, m * sizeof(OSQPInt));
  //
  d_predicate = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                nnz * sizeof(OSQPInt));
  d_compact_address = cl_allocate_mem(handle->context, CL_MEM_READ_WRITE,
                                      nnz * sizeof(OSQPInt));

  {

    cl_memcpy_d2d(d_row_predicate, d_rows, m * sizeof(OSQPInt));

    const cl_kernel kernel = handle->vector_init_abs_kernel;
    /*  */
    cl_int err =
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_row_predicate); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_row_predicate);
    err = clSetKernelArg(kernel, 2, sizeof(m), &m);

    const int number_of_blocks =
        (m / handle->deviceInformations[0].localworksize[0]) + 1;
    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }

  // Calculate new row numbering and get new number of rows
  inclusive_scan(d_row_predicate, d_new_row_number, m);

  if (m) {
    cl_memcpy_d2h_offset(&new_m, d_new_row_number, m - 1, sizeof(OSQPInt));
  } else {
    (*Ared) = (csr *)c_calloc(1, sizeof(csr));
    (*Ared)->n = n;

    (*Aredt) = (csr *)c_calloc(1, sizeof(csr));
    (*Aredt)->m = n;

    return;
  }

  {
    const cl_kernel kernel = handle->predicate_generator_kernel;
    /*  */
    cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_row_predicate);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_predicate);

    err = clSetKernelArg(kernel, 3, sizeof(nnz), &nnz);

    const int number_of_blocks =
        (nnz / handle->deviceInformations[0].localworksize[0]) + 1;
    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }

  // Get array offset for compacting and new nnz
  inclusive_scan(d_predicate, d_compact_address, nnz);

  OSQPInt nnz_new = 0;
  if (nnz) {
    cl_memcpy_d2h_offset(&nnz_new, d_compact_address, nnz - 1, sizeof(OSQPInt));
  }

  {
    // allocate new matrix (2 -> allocate row indices as well)
    (*Ared) = csr_alloc(new_m, n, nnz_new, 2);

    cl_kernel kernel = handle->compact_rows;
    /*  */
    cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(*Ared)->cl_row_ind);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_new_row_number);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_predicate);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_compact_address);

    err = clSetKernelArg(kernel, 5, sizeof(nnz), &nnz);

    const int number_of_blocks =
        (nnz / handle->deviceInformations[0].localworksize[0]) + 1;
    size_t workgroup[1] = {number_of_blocks};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);

    kernel = handle->compact;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->cl_col_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(*Ared)->cl_col_ind);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_predicate);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_compact_address);
    err = clSetKernelArg(kernel, 4, sizeof(nnz), &nnz);

    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A->cl_val); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(*Ared)->cl_val);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_predicate);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_compact_address);
    err = clSetKernelArg(kernel, 4, sizeof(nnz), &nnz);
    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                                 &local[0], 0, 0, 0);
  }

  // Generate row pointer
  compress_row_ind(*Ared);

  // We first make a copy of Ared
  *Aredt = csr_alloc(new_m, n, nnz_new, 1);

  cl_memcpy_d2d((*Aredt)->cl_val, (*Ared)->cl_val, nnz_new * sizeof(OSQPFloat));
  cl_memcpy_d2d((*Aredt)->cl_row_ptr, (*Ared)->cl_row_ptr,
                (new_m + 1) * sizeof(OSQPInt));
  cl_memcpy_d2d((*Aredt)->cl_col_ind, (*Ared)->cl_col_ind,
                nnz_new * sizeof(OSQPInt));

  cl_mem d_A_to_At_ind;
  csr_transpose(*Aredt, &d_A_to_At_ind);

  csr_expand_row_ind(*Ared);
  csr_expand_row_ind(*Aredt);

  init_SpMV_interface(*Ared);
  init_SpMV_interface(*Aredt);

  if (d_A_to_At_ind) {
    cl_free(d_A_to_At_ind);
  }
  cl_free(d_predicate);
  cl_free(d_compact_address);
  cl_free(d_row_predicate);
  cl_free(d_new_row_number);
}

void inclusive_scan(const cl_mem A, cl_mem result, OSQPInt n) {
  cl_kernel kernel = handle->inclusive_scan_kernel;

  /*  */
  cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result);
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), NULL);

  const int number_of_blocks =
      (n / handle->deviceInformations[0].localworksize[0]) + 1;
  size_t workgroup[1] = {number_of_blocks};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &workgroup[0],
                               &local[0], 0, 0, 0);
}