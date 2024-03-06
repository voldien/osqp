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

#include "include/opencl_lin_alg.h"

// #include "csr_type.h"
#include "glob_opts.h"
#include "opencl_alloc.h"
#include "opencl_helper.h"
#include "osqp_api_types.h"
#include <CL/cl.h>
#include <string.h>

extern OpenCL_Handle_t *handle;

void scatter(OSQPFloat *out, const OSQPFloat *in, const OSQPInt *ind,
             OSQPInt n) {

  int err;
  const cl_kernel kernel = handle->scatter_kernel;

  /*  */
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &out); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &ind);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / 64) + 1};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);
}

/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cl_vecf_create(cl_mem *vec, const OSQPFloat *d_x, OSQPInt n) {

  const unsigned int unalignedSize = n * sizeof(OSQPFloat);
  const unsigned int alignedSize =
      align(unalignedSize, handle->deviceInformations[0].alignedSize);
  /*  */
  *vec = cl_allocate_mem(handle->context,
                         CL_MEM_READ_WRITE |
                             CL_MEM_ALLOC_HOST_PTR // |
                                                   // CL_MEM_HOST_READ_ONLY
                         ,
                         alignedSize);
  if (d_x) {
    cl_memcpy_h2d(*vec, d_x, unalignedSize);
  }
}

void cl_veci_create(cl_mem *vec, const OSQPInt *d_x, OSQPInt n) {
  const unsigned int unalignedSize = n * sizeof(OSQPInt);
  const unsigned int alignedSize =
      align(unalignedSize, handle->deviceInformations[0].alignedSize);
  /*  */
  *vec = cl_allocate_mem(handle->context,
                         CL_MEM_READ_WRITE |
                             CL_MEM_ALLOC_HOST_PTR // |
                                                   // CL_MEM_HOST_READ_ONLY
                         ,
                         alignedSize);
  if (d_x) {
    cl_memcpy_h2d(*vec, d_x, unalignedSize);
  }
}

void cl_vec_destroy(cl_mem vec) {
  if (vec) {
    cl_int error = clReleaseMemObject(vec);
  }
}

void cl_vec_copy_h2d(cl_mem d_y, const OSQPFloat *h_x, OSQPInt n) {
  cl_memcpy_h2d(d_y, h_x, n * sizeof(OSQPFloat));
}

void cl_vec_copy_d2h(OSQPFloat *h_y, const cl_mem d_x, OSQPInt n) {

  cl_int err;
  memcpy(h_y, d_x, n * sizeof(OSQPFloat));
  return;
  int *ptr =
      (int *)clEnqueueMapBuffer(handle->queue, d_x, CL_TRUE, CL_MAP_WRITE, 0,
                                n * sizeof(OSQPFloat), 0, NULL, NULL, &err);
  int i;
  memcpy(h_y, ptr, n * sizeof(OSQPFloat));
  err = clEnqueueUnmapMemObject(handle->queue, d_x, ptr, 0, NULL, NULL);
}

void cl_vec_int_copy_h2d(cl_mem d_y, const OSQPInt *h_x, OSQPInt n) {

  cl_memcpy_h2d(d_y, h_x, n * sizeof(OSQPInt));
}

void cl_vec_int_copy_d2h(OSQPInt *h_y, const cl_mem d_x, OSQPInt n) {

  cl_memcpy_d2h(h_y, d_x, n * sizeof(OSQPInt));
}

void cl_vec_set_sc(cl_mem d_a, OSQPFloat sc, OSQPInt n) {
  cl_int err;

  const cl_kernel kernel = handle->vec_set_sc_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(sc), &sc);
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);
}

void cl_vec_set_sc_cond(cl_mem d_a, const cl_mem d_test, OSQPFloat sc_if_neg,
                        OSQPFloat sc_if_zero, OSQPFloat sc_if_pos, OSQPInt n) {

  cl_int err;

  const cl_kernel kernel = handle->vec_set_sc_cond_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_test);
  err = clSetKernelArg(kernel, 2, sizeof(sc_if_neg), &sc_if_neg);
  err = clSetKernelArg(kernel, 3, sizeof(sc_if_zero), &sc_if_zero);
  err = clSetKernelArg(kernel, 4, sizeof(sc_if_pos), &sc_if_pos);
  err = clSetKernelArg(kernel, 5, sizeof(n), &n);

  size_t workgroup[1] = {(n / 64) + 1};
  err = clEnqueueNDRangeKernel(handle->queue, handle->vec_set_sc_cond_kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);
}

void cl_vec_mult_sc(cl_mem d_a, OSQPFloat sc, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_mult_sc;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(sc), &sc);
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_add_scaled(cl_mem d_x, const cl_mem d_a, const cl_mem d_b,
                       OSQPFloat sca, OSQPFloat scb, OSQPInt n) {

  const cl_kernel kernel = handle->vec_add_scaled_kernel;

  cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);        /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_x);        /**/
  err = clSetKernelArg(kernel, 3, sizeof(sca), &sca);
  err = clSetKernelArg(kernel, 4, sizeof(scb), &scb);

  err = clSetKernelArg(kernel, 5, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  // TODO: optimize based on scaler.
}

void cl_vec_add_scaled3(cl_mem d_x, const cl_mem d_a, const cl_mem d_b,
                        const cl_mem d_c, OSQPFloat sca, OSQPFloat scb,
                        OSQPFloat scc, OSQPInt n) {

  const cl_kernel kernel = handle->vec_add_scaled3_kernel;

  cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);        /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);        /**/
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_x);        /**/
  err = clSetKernelArg(kernel, 4, sizeof(sca), &sca);
  err = clSetKernelArg(kernel, 5, sizeof(scb), &scb);
  err = clSetKernelArg(kernel, 6, sizeof(scb), &scc);

  err = clSetKernelArg(kernel, 7, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  // TODO: optimize based on scaler.
}

void cl_vec_norm_inf(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res) {

  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                      sizeof(*h_res));

  const cl_kernel kernel = handle->vec_norm_inf_kernel;

  cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x); /**/

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp); /**/
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);        /**/

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(*h_res));
  cl_free(tmp);
}

void cl_vec_norm_2(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res) {

  cl_vec_norm_1(d_x, n, h_res);
  *h_res = c_sqrt(*h_res);
}

void cl_vec_scaled_norm_inf(const cl_mem d_S, const cl_mem d_v, OSQPInt n,
                            OSQPFloat *h_res) {

  cl_mem d_v_scaled =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                      n * sizeof(OSQPFloat));

  /* d_v_scaled = d_S * d_v */
  cl_vec_ew_prod(d_v_scaled, d_S, d_v, n);

  /* (*h_res) = |d_v_scaled|_inf */
  cl_vec_norm_inf(d_v_scaled, n, h_res);

  cl_free(d_v_scaled);
}

void cl_vec_diff_norm_inf(const cl_mem d_a, const cl_mem d_b, OSQPInt n,
                          OSQPFloat *h_res) {

  cl_mem d_diff =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, n * sizeof(OSQPFloat));

  /* d_diff = d_a - d_b */
  cl_vec_add_scaled(d_diff, d_a, d_b, 1.0, -1.0, n);

  /* (*h_res) = |d_diff|_inf */
  cl_vec_norm_inf(d_diff, n, h_res);

  cl_free(d_diff);
}

void cl_vec_norm_1(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res) {

  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                      sizeof(OSQPFloat));

  const cl_kernel kernel = handle->vec_sum_kernel;
  cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x); /**/

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp); /**/
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, &local[0],
                               &workgroup[0], 0, 0, 0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(OSQPFloat));
  cl_free(tmp);
}

void cl_vec_prod(const cl_mem d_a, const cl_mem d_b, OSQPInt n,
                 OSQPFloat *h_res) {

  const cl_kernel kernel = handle->vec_prod_kernel;
  cl_int err;
  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, sizeof(OSQPFloat));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b); /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(OSQPFloat));
  cl_free(tmp);
}

void cl_vec_prod_signed(const cl_mem d_a, const cl_mem d_b, OSQPInt sign,
                        OSQPInt n, OSQPFloat *h_res) {

  cl_kernel kernel;
  if (sign == 1) {
    kernel = handle->vec_prod_pos_kernel;
  } else if (sign == -1) {
    kernel = handle->vec_prod_neg_kernel;
  } else {
    cl_vec_prod(d_a, d_b, n, h_res);
    return;
  }

  cl_int err;
  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, sizeof(OSQPFloat));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b); /**/
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(OSQPFloat));
  cl_free(tmp);
}

void cl_vec_ew_prod(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_ew_prod_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};

  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1,
                               NULL /*&localworksize[0]*/, &workgroup[0], 0, 0,
                               0, 0);
}

void cl_vec_eq(const cl_mem a, const cl_mem b, OSQPFloat tol, OSQPInt n,
               OSQPInt *h_res) {

  cl_int err;
  const cl_kernel kernel = handle->vec_eq_kernel;

  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, sizeof(OSQPInt));
  cl_memset(tmp, 1, 0, sizeof(*h_res));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
  err = clSetKernelArg(kernel, 2, sizeof(tol), &tol);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel, 4, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(*h_res));
  cl_free(tmp);
  *h_res = (unsigned int)*h_res;
}

void cl_vec_leq(const cl_mem d_l, const cl_mem d_u, OSQPInt n, OSQPInt *h_res) {

  cl_int err;
  const cl_kernel kernel = handle->vec_leq_kernel;

  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, sizeof(OSQPInt));
  cl_memset(tmp, 1, 0, sizeof(*h_res));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_l);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_u);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(*h_res));
  cl_free(tmp);
  *h_res = (unsigned int)*h_res;
}

void cl_vec_bound(cl_mem d_x, const cl_mem d_z, const cl_mem d_l,
                  const cl_mem d_u, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_bound_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_z);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_l);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_u);

  err = clSetKernelArg(kernel, 4, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_project_polar_reccone(cl_mem d_y, const cl_mem d_l,
                                  const cl_mem d_u, OSQPFloat infval,
                                  OSQPInt n) {
  cl_int err;
  const cl_kernel kernel = handle->vec_project_polar_reccone_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_y); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_l);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_u);
  err = clSetKernelArg(kernel, 3, sizeof(infval), &infval);

  err = clSetKernelArg(kernel, 4, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_in_reccone(const cl_mem d_y, const cl_mem d_l, const cl_mem d_u,
                       OSQPFloat infval, OSQPFloat tol, OSQPInt n,
                       OSQPInt *h_res) {

  cl_int err;
  const cl_kernel kernel = handle->vec_in_reccone_kernel;

  cl_mem tmp =
      cl_allocate_mem(handle->context, CL_MEM_READ_ONLY, sizeof(OSQPInt));
  cl_memset(tmp, 1, 0, sizeof(*h_res));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_y); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_l);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_u);
  err = clSetKernelArg(kernel, 3, sizeof(infval), &infval);
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &tmp);
  err = clSetKernelArg(kernel, 5, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  cl_memcpy_d2h(h_res, tmp, sizeof(*h_res));
  cl_free(tmp);
  *h_res = (unsigned int)*h_res;
}

void cl_vec_reciprocal(cl_mem d_b, const cl_mem d_a, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_reciprocal_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_b); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 2, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_sqrt(cl_mem d_a, OSQPInt n) {
  cl_int err;
  const cl_kernel kernel = handle->vec_sqrt_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a); /**/
  err = clSetKernelArg(kernel, 1, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_max(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_max_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_min(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n) {

  cl_int err;
  const cl_kernel kernel = handle->vec_min_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err = clSetKernelArg(kernel, 3, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_bounds_type(cl_mem d_iseq, const cl_mem d_l, const cl_mem d_u,
                        OSQPFloat infval, OSQPFloat tol, OSQPInt n,
                        OSQPInt *h_has_changed) {

  cl_int err;
  const cl_kernel kernel = handle->vec_bounds_type_kernel;

  cl_mem tmp = cl_allocate_calloc_mem(handle->context, CL_MEM_READ_ONLY,
                                      sizeof(*h_has_changed));

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_iseq); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_l);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_u);
  err = clSetKernelArg(kernel, 3, sizeof(infval), &infval);
  err = clSetKernelArg(kernel, 4, sizeof(tol), &tol);
  err = clSetKernelArg(kernel, 5, sizeof(h_has_changed), tmp);

  err = clSetKernelArg(kernel, 6, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  cl_memcpy_d2h(h_has_changed, tmp, sizeof(*h_has_changed));
  cl_free(tmp);
}

void cl_vec_set_sc_if_lt(cl_mem d_x, const cl_mem d_z, OSQPFloat testval,
                         OSQPFloat newval, OSQPInt n) {
  cl_int err;
  const cl_kernel kernel = handle->vec_set_sc_if_lt_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_z);
  err = clSetKernelArg(kernel, 2, sizeof(testval), &testval);
  err = clSetKernelArg(kernel, 3, sizeof(newval), &newval);
  err = clSetKernelArg(kernel, 4, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_set_sc_if_gt(cl_mem d_x, const cl_mem d_z, OSQPFloat testval,
                         OSQPFloat newval, OSQPInt n) {
  cl_int err;
  const cl_kernel kernel = handle->vec_set_sc_if_gt_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_z);
  err = clSetKernelArg(kernel, 2, sizeof(testval), &testval);
  err = clSetKernelArg(kernel, 3, sizeof(newval), &newval);
  err = clSetKernelArg(kernel, 4, sizeof(n), &n);

  size_t workgroup[1] = {(n / handle->deviceInformations[0].localworksize[0]) +
                         1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_vec_gather(OSQPInt nnz, const OSQPFloat *d_y, OSQPFloat *d_xVal,
                   const OSQPInt *d_xInd) {

  // TODO: impl
}

void cl_mat_mult_sc(csr *S, csr *At, OSQPFloat sc) {

  cl_vec_mult_sc(S->mainbuffer, sc, S->nnz);
  if (At) {
    /* Update At as well */
    cl_vec_mult_sc(At->mainbuffer, sc, S->nnz);
  }
}

void cl_mat_lmult_diag(csr *S, csr *At, const cl_mem d_diag) {

  return;
  const OSQPInt nnz = S->nnz;

  cl_int err;
  const cl_kernel kernel = handle->mat_lmult_diag_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &S->cl_row_ind); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_diag);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &S->cl_val);
  err = clSetKernelArg(kernel, 3, sizeof(nnz), &nnz);

  size_t workgroup[1] = {
      (nnz / handle->deviceInformations[0].localworksize[0]) / 8 + 1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  if (At) {
    //   /* Multiply At from right */
    err = clFlush(handle->queue);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &At->cl_col_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_diag);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &At->cl_val);
    err = clSetKernelArg(kernel, 3, sizeof(nnz), &nnz);

    size_t workgroup[1] = {
        (nnz / handle->deviceInformations[0].localworksize[0]) + 1};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                                 &workgroup[0], 0, 0, 0);
  }
}

void cl_mat_rmult_diag(csr *S, csr *At, const cl_mem d_diag) {

  return;
  // TODO: Impl
  const OSQPInt nnz = S->nnz;

  cl_int err;
  const cl_kernel kernel = handle->mat_rmult_diag_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &S->cl_col_ind); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_diag);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &S->cl_val);
  err = clSetKernelArg(kernel, 3, sizeof(nnz), &nnz);

  size_t workgroup[1] = {
      (nnz / handle->deviceInformations[0].localworksize[0]) / 8 + 1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);

  if (At) {
    //   /* Multiply At from right */
    err = clFlush(handle->queue);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &At->cl_row_ind); /**/
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_diag);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &At->cl_val);
    err = clSetKernelArg(kernel, 3, sizeof(nnz), &nnz);

    size_t workgroup[1] = {
        (nnz / handle->deviceInformations[0].localworksize[0]) + 1};
    size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
    err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                                 &workgroup[0], 0, 0, 0);
  }
}

void cl_mat_rmult_diag_new(const csr *S, cl_mem d_buffer, const cl_mem d_diag) {
  return;
  const OSQPInt nnz = S->nnz;
  // TODO: Impl

  cl_int err;
  const cl_kernel kernel = handle->mat_rmult_diag_new_kernel;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &S->cl_col_ind); /**/
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_diag);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &S->cl_val);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_buffer);
  err = clSetKernelArg(kernel, 4, sizeof(nnz), &nnz);

  size_t workgroup[1] = {
      (nnz / handle->deviceInformations[0].localworksize[0]) / 8 + 1};
  size_t local[1] = {handle->deviceInformations[0].localworksize[0]};
  err = clEnqueueNDRangeKernel(handle->queue, kernel, 1, NULL, &local[0],
                               &workgroup[0], 0, 0, 0);
}

void cl_mat_Axpy(const csr *A, const OSQPVectorf *vecx, OSQPVectorf *vecy,
                 OSQPFloat alpha, OSQPFloat beta) {

  cl_int err;
  err = clSetKernelArg(handle->Axpy_mat_kernel, 0, sizeof(cl_mem),
                       &A->mainbuffer); /**/
  err = clSetKernelArg(handle->Axpy_mat_kernel, 1, sizeof(cl_mem),
                       &vecx->cl_vec); /**/
  err = clSetKernelArg(handle->Axpy_mat_kernel, 2, sizeof(cl_mem),
                       &vecy->cl_vec); /**/

  err = clSetKernelArg(handle->Axpy_mat_kernel, 3, sizeof(alpha), &alpha);
  err = clSetKernelArg(handle->Axpy_mat_kernel, 4, sizeof(beta), &beta);
  err = clSetKernelArg(handle->Axpy_mat_kernel, 5, sizeof(OSQPInt), &A->n);
  err = clSetKernelArg(handle->Axpy_mat_kernel, 6, sizeof(OSQPInt), &A->m);

  cl_mem tmp = cl_allocate_mem(handle->context, CL_MEM_WRITE_ONLY,
                               (A->m) * (vecy->length) * sizeof(OSQPFloat));
  err = clSetKernelArg(handle->Axpy_mat_kernel, 7, sizeof(cl_mem), &tmp);

  size_t workgroup[2] = {
      ((A->m) / handle->deviceInformations[0].localworksize[0]) + 1,
      ((vecy->length) / handle->deviceInformations[0].localworksize[1]) + 1};
  size_t local[2] = {A->m, vecy->length};

  err = clEnqueueNDRangeKernel(handle->queue, handle->Axpy_mat_kernel, 2, NULL,
                               &local[0], &workgroup[0], 0, 0, 0);

  clFlush(handle->queue);
  cl_memcpy_d2d(vecy->cl_vec, tmp, vecy->length * sizeof(OSQPFloat));
  cl_free(tmp);
}

void cl_mat_row_norm_inf(const csr *S, OSQPFloat *d_res) {

  OSQPInt nnz = S->nnz;
  OSQPInt num_rows = S->m;

  if (nnz == 0) {
    return;
  }
  // TODO: Impl
}