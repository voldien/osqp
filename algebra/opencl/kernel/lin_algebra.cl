// #pragma OPENCL EXTENSION cl_khr_il_program : enable
// #pragma OPENCL EXTENSION cl_khr_spir : enable
// #pragma OPENCL EXTENSION cl_ext_float_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define OSQP_USE_FLOAT

#ifdef OSQP_USE_FLOAT
typedef float OSQPFloat;
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
typedef double OSQPFloat;
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef OSQP_USE_FLOAT
typedef long OSQPInt;
#else
typedef int OSQPInt;
#endif

__kernel void vec_norm_inf_kernel(__global const OSQPFloat *A,
                                  __global OSQPFloat *result, const OSQPInt n) {
  __local OSQPFloat a[128];

  // size_t warp_size = 32;
  const size_t sub_group_size = get_sub_group_size();

  const size_t local_id_0 = get_local_id(0);
  const size_t local_size_0 = get_local_size(0);

  const size_t warp_num = local_id_0 / sub_group_size;
  const size_t warp_lane = local_id_0 % sub_group_size;

  OSQPFloat updates0 = 0;

  for (size_t idx = local_id_0; idx < n; idx += local_size_0) {
    const OSQPFloat tmp = fabs(A[idx]);
    updates0 = fmax(updates0, tmp);
  }

  updates0 = sub_group_reduce_max(updates0);
  if (warp_lane == 0) {
    a[warp_num] = updates0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (warp_num == 0) {
    updates0 = a[warp_lane];
    updates0 = sub_group_reduce_max(updates0);
    if (warp_lane == 0) {
      result[0] = updates0;
    }
  }
}

__kernel void Axpy_mat_kernel(__global const OSQPFloat *A,
                              __global const OSQPFloat *x, /*  */
                              __global const OSQPFloat *y, OSQPFloat alpha,
                              const OSQPFloat beta, const OSQPInt n,
                              const OSQPInt m, __global OSQPFloat *data) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_stride = get_num_groups(0) * get_local_size(0);

  int k;
  int i = get_global_id(0);
  int j = get_global_id(1);
  int Order0 = get_global_size(0);
  int Order1 = get_global_size(1);

  OSQPFloat tmp = (OSQPFloat)0.0;

  for (k = 0; k < Order0; k++) {
    tmp += A[i * Order0 + k] * x[k * Order1 + j];

    data[i * Order0 + j] = alpha * tmp + beta * y[j];
  }
}

// [OpenCL 2.0 or greater] or support subgroup extension
__kernel void vec_sum_kernel(__constant const OSQPFloat *arr,
                             __global OSQPFloat *sum, OSQPInt n) {
  __local OSQPFloat a[32];

  // size_t warp_size = 32;
  const size_t sub_group_size = get_sub_group_size();

  const size_t local_id_0 = get_local_id(0);
  const size_t local_size_0 = get_local_size(0);

  const size_t warp_num = local_id_0 / sub_group_size;
  const size_t warp_lane = local_id_0 % sub_group_size;

  OSQPFloat updates0 = 0.0;

  for (size_t idx = local_id_0; idx < n; idx += local_size_0) {
    updates0 += arr[idx];
  }

  // printf("%i, %i, %f\n", sub_group_size, 0, updates0);

  updates0 = sub_group_reduce_add(updates0);
  if (warp_lane == 0) {
    a[warp_num] = updates0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (warp_num == 0) {
    updates0 = a[warp_lane];
    updates0 = sub_group_reduce_add(updates0);
    if (warp_lane == 0) {
      sum[0] = updates0;
    }
  }
}

__kernel void vec_set_sc_kernel(__global OSQPFloat *a, OSQPFloat sc,
                                OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    a[i] = sc;
  }
}

__kernel void vec_mult_sc_kernel(__global OSQPFloat *a, OSQPFloat sc,
                                 OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    a[i] *= sc;
  }
}

__kernel void vec_add_scaled_kernel(__global const OSQPFloat *a,
                                    __global const OSQPFloat *b,
                                    __global OSQPFloat *res, const OSQPFloat ac,
                                    const OSQPFloat bc, const OSQPInt n) {
  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    res[i] = a[i] * ac + b[i] * bc;
  }
}

__kernel void vec_add_scaled3_kernel(__global const OSQPFloat *a,
                                     __global const OSQPFloat *b,
                                     __global const OSQPFloat *c,
                                     __global OSQPFloat *res,
                                     const OSQPFloat ac, const OSQPFloat bc,
                                     const OSQPFloat cc, const OSQPInt n) {
  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    res[i] = a[i] * ac + b[i] * bc + c[i] * cc;
  }
}

__kernel void vec_set_sc_cond_kernel(__global OSQPFloat *a,
                                     __global const OSQPInt *test,
                                     OSQPFloat sc_if_neg, OSQPFloat sc_if_zero,
                                     OSQPFloat sc_if_pos, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {

    if (test[i] == 0) {
      a[i] = sc_if_zero;
    } else if (test[i] > 0) {
      a[i] = sc_if_pos;
    } else {
      a[i] = sc_if_neg;
    }
  }
}

__kernel void vec_prod_kernel(__global const OSQPFloat *a,
                              __global const OSQPFloat *b,
                              __global OSQPFloat *res, OSQPInt n) {
  __local OSQPFloat a_cache[32];
  // size_t warp_size = 32;
  const size_t sub_group_size = get_sub_group_size();

  const size_t local_id_0 = get_local_id(0);
  const size_t local_size_0 = get_local_size(0);

  const size_t warp_num = local_id_0 / sub_group_size;
  const size_t warp_lane = local_id_0 % sub_group_size;

  OSQPFloat updates0 = 0.0;

  for (size_t idx = local_id_0; idx < n; idx += local_size_0) {
    updates0 += a[idx] * b[idx];
  }

  updates0 = sub_group_reduce_add(updates0);
  if (warp_lane == 0) {
    a_cache[warp_num] = updates0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (warp_num == 0) {
    updates0 = a_cache[warp_lane];
    updates0 = sub_group_reduce_add(updates0);
    if (warp_lane == 0) {
      res[0] = updates0;
    }
  }
}

__kernel void vec_prod_pos_kernel(__global const OSQPFloat *a,
                                  __global const OSQPFloat *b,
                                  __global OSQPFloat *res, OSQPInt n) {
  __local OSQPFloat a_cache[32];
  // size_t warp_size = 32;
  const size_t sub_group_size = get_sub_group_size();

  const size_t local_id_0 = get_local_id(0);
  const size_t local_size_0 = get_local_size(0);

  const size_t warp_num = local_id_0 / sub_group_size;
  const size_t warp_lane = local_id_0 % sub_group_size;

  OSQPFloat updates0 = 0.0;

  for (size_t idx = local_id_0; idx < n; idx += local_size_0) {
    updates0 += a[idx] * max(b[idx], (OSQPFloat)0.0);
  }

  updates0 = sub_group_reduce_add(updates0);
  if (warp_lane == 0) {
    a_cache[warp_num] = updates0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (warp_num == 0) {
    updates0 = a_cache[warp_lane];
    updates0 = sub_group_reduce_add(updates0);
    if (warp_lane == 0) {
      res[0] = updates0;
    }
  }
}

__kernel void vec_prod_neg_kernel(__global const OSQPFloat *a,
                                  __global const OSQPFloat *b,
                                  __global OSQPFloat *res, OSQPInt n) {
  __local OSQPFloat a_cache[32];
  // size_t warp_size = 32;
  const size_t sub_group_size = get_sub_group_size();

  const size_t local_id_0 = get_local_id(0);
  const size_t local_size_0 = get_local_size(0);

  const size_t warp_num = local_id_0 / sub_group_size;
  const size_t warp_lane = local_id_0 % sub_group_size;

  OSQPFloat updates0 = 0.0;

  for (size_t idx = local_id_0; idx < n; idx += local_size_0) {
    updates0 += a[idx] * min(b[idx], (OSQPFloat)0.0);
  }

  updates0 = sub_group_reduce_add(updates0);
  if (warp_lane == 0) {
    a_cache[warp_num] = updates0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (warp_num == 0) {
    updates0 = a_cache[warp_lane];
    updates0 = sub_group_reduce_add(updates0);
    if (warp_lane == 0) {
      res[0] = updates0;
    }
  }
}

__kernel void vec_ew_prod_kernel(__global OSQPFloat *c,
                                 __global const OSQPFloat *a,
                                 __global const OSQPFloat *b, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = a[i] * b[i];
  }
}

__kernel void vec_eq_kernel(__global const OSQPFloat *a,
                            __global const OSQPFloat *b, OSQPFloat tol,
                            __global OSQPInt *res, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  *res = 1;

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if (fabs(a[i] - b[i]) > tol) {
      *res = 0;
      break;
    }
  }
}

__kernel void vec_leq_kernel(__global const OSQPFloat *l,
                             __global const OSQPFloat *u,
                             __global volatile OSQPInt *res, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if (l[i] > u[i]) {
      atomic_and((__global volatile uint *)&res[0], 0);
      atomic_and((__global volatile uint *)&res[1], 0);
    }
  }
}

__kernel void vec_bound_kernel(__global OSQPFloat *x,
                               __global const OSQPFloat *z,
                               __global const OSQPFloat *l,
                               __global const OSQPFloat *u, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = min(max(z[i], l[i]), u[i]);
  }
}

__kernel void vec_project_polar_reccone_kernel(__global OSQPFloat *y,
                                               __global const OSQPFloat *l,
                                               __global const OSQPFloat *u,
                                               OSQPFloat infval, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if (u[i] > +infval) {
      if (l[i] < -infval) {
        /* Both bounds infinite */
        y[i] = 0.0;
      } else {
        /* Only upper bound infinite */
        y[i] = min(y[i], (OSQPFloat)0.0);
      }
    } else if (l[i] < -infval) {
      /* Only lower bound infinite */
      y[i] = max(y[i], (OSQPFloat)0.0);
    }
  }
}

__kernel void vec_in_reccone_kernel(__global const OSQPFloat *y,
                                    __global const OSQPFloat *l,
                                    __global const OSQPFloat *u,
                                    OSQPFloat infval, OSQPFloat tol,
                                    __global OSQPInt *res, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if ((u[i] < +infval && y[i] > +tol) || (l[i] > -infval && y[i] < -tol)) {
      atomic_and((__global volatile uint *)&res[0], 0);
      atomic_and((__global volatile uint *)&res[1], 0);
    }
  }
}

__kernel void vec_reciprocal_kernel(__global OSQPFloat *b,
                                    __global const OSQPFloat *a, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    b[i] = native_recip(a[i]);
  }
}

__kernel void vec_sqrt_kernel(__global OSQPFloat *a, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    a[i] = sqrt(a[i]);
  }
}

__kernel void vec_max_kernel(__global OSQPFloat *c, __global const OSQPFloat *a,
                             __global const OSQPFloat *b, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = max(a[i], b[i]);
  }
}

__kernel void vec_min_kernel(__global OSQPFloat *c, __global const OSQPFloat *a,
                             __global const OSQPFloat *b, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = min(a[i], b[i]);
  }
}

__kernel void vec_bounds_type_kernel(__global OSQPInt *iseq,
                                     __global const OSQPFloat *l,
                                     __global const OSQPFloat *u,
                                     OSQPFloat infval, OSQPFloat tol,
                                     volatile __global OSQPInt *has_changed,
                                     OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    if (u[i] - l[i] < tol) {
      /* Equality constraints */
      if (iseq[i] != 1) {
        iseq[i] = 1;
        atomic_or(has_changed, 1);
      }
    } else if ((l[i] < -infval) && (u[i] > infval)) {
      /* Loose bounds */
      if (iseq[i] != -1) {
        iseq[i] = -1;
        atomic_or(has_changed, 1);
      }
    } else {
      /* Inequality constraints */
      if (iseq[i] != 0) {
        iseq[i] = 0;
        atomic_or(has_changed, 1);
      }
    }
  }
}

__kernel void vec_set_sc_if_lt_kernel(__global OSQPFloat *x,
                                      __global const OSQPFloat *z,
                                      OSQPFloat testval, OSQPFloat newval,
                                      OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] < testval ? newval : z[i];
  }
}

__kernel void vec_set_sc_if_gt_kernel(__global OSQPFloat *x,
                                      __global const OSQPFloat *z,
                                      OSQPFloat testval, OSQPFloat newval,
                                      OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] > testval ? newval : z[i];
  }
}

__kernel void mat_lmult_diag_kernel(__global const OSQPInt *row_ind,
                                    __global const OSQPFloat *diag,
                                    __global OSQPFloat *data, OSQPInt nnz) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt row = row_ind[i];
    data[i] *= diag[row];
  }
}

__kernel void mat_rmult_diag_kernel(__global const OSQPInt *col_ind,
                                    __global const OSQPFloat *diag,
                                    __global OSQPFloat *data, OSQPInt nnz) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data[i] *= diag[column];
  }
}

__kernel void mat_rmult_diag_new_kernel(__global const OSQPInt *col_ind,
                                        __global const OSQPFloat *diag,
                                        __global const OSQPFloat *data_in,
                                        __global OSQPFloat *data_out,
                                        OSQPInt nnz) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data_out[i] = data_in[i] * diag[column];
  }
}

__kernel void vec_abs_kernel(__global OSQPFloat *a, const OSQPInt n) {

  OSQPInt i = get_local_id(0) + get_local_size(0) * get_group_id(0);

  if (i < n) {
#ifdef OSQP_USE_FLOAT
    a[i] = fabs(a[i]);
#else
    a[i] = fabs(a[i]);
#endif
  }
}

__kernel void scatter_kernel(__global OSQPFloat *out,
                             __global const OSQPFloat *in,
                             __global const OSQPInt *ind, OSQPInt n) {

  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);

  for (OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt j = ind[i];
    out[j] = in[i];
  }
}

__kernel void abs_kernel(__global const OSQPInt *index_one_based,
                         __global const OSQPFloat *d_x,
                         __global OSQPFloat *res) {
  (*res) = fabs((d_x[(*index_one_based) - 1]));
}

__kernel void dot_innerproduct_kernel(__global const OSQPFloat *a,
                                      __global const OSQPFloat *b,
                                      __global OSQPFloat *res, OSQPInt n) {
  OSQPInt idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
  OSQPInt grid_size = get_local_size(0) * get_num_groups(0);
}
