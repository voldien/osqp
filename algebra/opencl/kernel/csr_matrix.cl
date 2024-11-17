// #pragma OPENCL EXTENSION cl_khr_il_program : enable
// #pragma OPENCL EXTENSION cl_khr_spir : enable
#define OSQP_USE_FLOAT
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifdef OSQP_USE_FLOAT
typedef float OSQPFloat;

#else
typedef double OSQPFloat;
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef OSQP_USE_FLOAT
typedef int OSQPInt;
#else
typedef int OSQPInt;
#endif

__kernel void fill_full_matrix_kernel(
    __global OSQPInt *row_ind_out, __global OSQPInt *col_ind_out,
    __global OSQPInt *nnz_on_diag, __global OSQPInt *has_non_zero_diag_element,
    __global const OSQPInt *__restrict__ row_ind_in,
    __global const OSQPInt *__restrict__ col_ind_in, OSQPInt nnz, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt row = row_ind_in[i];
    OSQPInt column = col_ind_in[i];

    row_ind_out[i] = row;
    col_ind_out[i] = column;

    if (row == column) {
      has_non_zero_diag_element[row] = 1;
      row_ind_out[i + nnz] =
          column + n; /* dummy value for sorting and removal later on */
      col_ind_out[i + nnz] = row + n;
      atomic_add(&nnz_on_diag[0], 1);
      // atomic_add(&nnz_on_diag[1], 1);
      //   atomic_add(&nnz_on_diag[1], 1);
    } else {
      row_ind_out[i + nnz] = column;
      col_ind_out[i + nnz] = row;
    }
  }
}

/**
 * Insert elements at structural zeros on the diagonal of the sparse matrix
 * specified by row and column index (COO format). To keep a one-to-one memory
 * patern we add n new elements to the matrix. In case where there already is a
 * diagonal element we add a dummy entry. The dummy entries will be removed
 * later.
 */
__kernel void
add_diagonal_kernel(__global OSQPInt *row_ind, __global OSQPInt *col_ind,
                    __global const OSQPInt *has_non_zero_diag_element,
                    OSQPInt n, OSQPInt offset) {
  row_ind += offset;
  col_ind += offset;

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt row = idx; row < n; row += grid_size) {
    if (has_non_zero_diag_element[row] == 0) {
      row_ind[row] = row;
      col_ind[row] = row;
    } else {
      row_ind[row] = row + n; /* dummy value, for easy removal after sorting */
      col_ind[row] = row + n;
    }
  }
}

/*
 * Permutation in: (size n, range 2*nnz+n):
 *
 * Gathers from the following array to create the full matrix :
 *
 *       |P_lower->val|P_lower->val|zeros(n)|
 *
 *
 * Permutation out: (size n, range new_range)
 *
 * Gathers from the following array to create the full matrix :
 *
 *          |P_lower->val|zeros(1)|
 *
 *          | x[i] mod new_range    if x[i] <  2 * new_range
 * x[i] ->  | new_range             if x[i] >= 2 * new_range
 *
 */
__kernel void reduce_permutation_kernel(__global OSQPInt *permutation,
                                        OSQPInt new_range, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if (permutation[i] < 2 * new_range) {
      permutation[i] = permutation[i] % new_range;
    } else {
      permutation[i] =
          new_range; /* gets the 0 element at nnz+1 of the value array */
    }
  }
}

__kernel void get_diagonal_indices_kernel(__global const OSQPInt *row_ind,
                                          __global const OSQPInt *col_ind,
                                          const OSQPInt nnz,
                                          __global OSQPInt *diag_index) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt index = idx; index < nnz; index += grid_size) {
    OSQPInt row = row_ind[index];
    OSQPInt column = col_ind[index];

    if (row == column) {
      diag_index[row] = index;
    }
  }
}

__kernel void predicate_generator_kernel(__global const OSQPInt *row_ind,
                                         __global const OSQPInt *row_predicate,
                                         __global OSQPInt *predicate,
                                         const OSQPInt nnz) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_stride = get_num_groups(0) * get_local_size(0);

  for (OSQPInt i = idx; i < nnz; i += grid_stride) {
    OSQPInt row = row_ind[i];
    predicate[i] = row_predicate[row];
  }
}

__kernel void compact(__global const OSQPFloat *data_in,
                      __global OSQPFloat *data_out, __global OSQPInt *predicate,
                      __global OSQPInt *scatter_addres, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);

  if (idx < n) {
    if (predicate[idx]) {
      int write_ind = scatter_addres[idx] - 1;
      data_out[write_ind] = data_in[idx];
    }
  }
}

__kernel void compact_rows(__global const OSQPInt *row_ind,
                           __global OSQPInt *data_out,
                           __global OSQPInt *new_row_number,
                           __global OSQPInt *predicate,
                           __global OSQPInt *scatter_addres, OSQPInt n) {

  int idx = get_local_id(0) + get_local_size(0) * get_group_id(0);

  if (idx < n) {
    if (predicate[idx]) {
      OSQPInt write_ind = scatter_addres[idx] - 1;
      OSQPInt row = row_ind[idx];
      data_out[write_ind] = new_row_number[row] - 1;
    }
  }
}

__kernel void vector_init_abs_kernel(__global const OSQPInt *a,
                                     __global OSQPInt *b, OSQPInt n) {

  OSQPInt i = get_local_id(0) + get_local_size(0) * get_group_id(0);

  if (i < n) {
    b[i] = abs(a[i]);
  }
}

__kernel void csr_eq_kernel(__global const OSQPInt *A_row_ptr,
                            __global const OSQPInt *A_col_ind,
                            __global const OSQPFloat *A_val,
                            __global const OSQPInt *B_row_ptr,
                            __global const OSQPInt *B_col_ind,
                            __global const OSQPFloat *B_val, const OSQPInt m,
                            const OSQPFloat tol, __global OSQPInt *res) {
  OSQPInt i = 0;
  OSQPInt j = 0;
  OSQPFloat diff = 0.0;

  *res = 1;

  for (j = 0; j < m; j++) { // Cycle over rows j
    // if row pointer of next row does not coincide, they are not equal
    // NB: first row always has A->p[0] = B->p[0] = 0 by construction.
    if (A_row_ptr[j + 1] != B_row_ptr[j + 1]) {
      *res = 0;
      return;
    }

    for (i = A_row_ptr[j]; i < A_row_ptr[j + 1];
         i++) {                           // Cycle columns i in row j
      if (A_col_ind[i] != B_col_ind[i]) { // Different column indices
        *res = 0;
        return;
      }

      diff = fabs(A_val[i] - B_val[i]);
      if (diff > tol) { // The actual matrix values are different
        *res = 0;
        return;
      }
    }
  }
}

__kernel void mat_mul(__global OSQPFloat *A, __global OSQPFloat *B,
                      __global OSQPFloat *C) {

  int k;
  int i = get_global_id(0);
  int j = get_global_id(1);
  int Order = get_global_size(0);

  OSQPFloat tmp = 0.0f;
  for (k = 0; k < Order; k++) {
    tmp += A[i * Order + k] * B[k * Order + j];
    // C(i, j) = sum(over k) A(i,k) * B(k,j)!
    C[i * Order + j] = tmp;
  }
}

void matrix_multiplication(OSQPFloat *A, OSQPFloat *B, OSQPFloat *O, OSQPInt M,
                           OSQPInt N, OSQPInt K) {
  int i = get_global_id(0);
  int k = get_global_id(0);
  // int  = get_global_id(0);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {

      OSQPFloat acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[k * M + m] * B[n * K + k];
      }

      O[n * M + m] = acc;
    }
  }
}
#define WORKGROUP_SIZE 256

__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1))) void
inclusive_scan_kernel(__global const OSQPInt *A, __global OSQPInt *result,
                      OSQPInt n, __local OSQPInt *l_Data) {

  OSQPInt prefix_sum_val = work_group_scan_inclusive_add(A[get_local_id(0)]);
  // Load data
  // OSQPInt idata4 = A[get_global_id(0)];

  // Calculate exclusive scan
  // OSQPInt odata4 = scan4Inclusive(idata4, l_Data, n);

  // Write back
  result[get_local_id(0)] = prefix_sum_val;
}
#pragma OPENCL EXTENSION cl_amd_printf : enable
__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1))) __kernel void
offsets_to_indices_kernel(const OSQPInt num_rows,
                          __global const OSQPInt *offsets,
                          __global OSQPInt *indices) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_stride = get_num_groups(0) * get_local_size(0);

  for (OSQPInt i = idx; i < num_rows; i += grid_stride) {
    const OSQPInt row_start = offsets[i];
    const OSQPInt row_end = offsets[i + 1];

    for (OSQPInt j = row_start + 1; j < row_end; j += 1) {
      indices[j] = i;
    }
  }

  //const int SUBWAVE_SIZE = 1;
//
  //const OSQPInt global_id = get_global_id(0); // global workitem id
  //const OSQPInt local_id = get_local_id(0);   // local workitem id
  //const OSQPInt thread_lane = local_id & (SUBWAVE_SIZE - 1);
  //const OSQPInt vector_id = global_id / SUBWAVE_SIZE; // global vector id
  //const OSQPInt num_vectors = get_global_size(0) / SUBWAVE_SIZE;
//
  //for (OSQPInt row = vector_id; row < num_rows; row += num_vectors) {
//
  //  const OSQPInt row_start = offsets[row];
  //  const OSQPInt row_end = offsets[row + 1];
//
  //  for (OSQPInt j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
  //    indices[j] = row;
  //}
}
