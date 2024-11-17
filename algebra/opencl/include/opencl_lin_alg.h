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

#ifndef OPENCL_LIN_ALG_H
#define OPENCL_LIN_ALG_H

#include "algebra_types.h"
#include "algebra_vector.h"
#include "opencl_helper.h"

/*******************************************************************************
 *                           Vector Functions                                  *
 *******************************************************************************/

void cl_vecf_create(cl_mem *vec, const OSQPFloat *d_x, OSQPInt n);
void cl_veci_create(cl_mem *vec, const OSQPInt *d_x, OSQPInt n);

void cl_vec_destroy(cl_mem vec);

/*
 * d_y[i] = h_x[i] for i in [0,n-1]
 */
void cl_vec_copy_h2d(cl_mem d_y, const OSQPFloat *h_x, OSQPInt n);

/*
 * h_y[i] = d_x[i] for i in [0,n-1]
 */
void cl_vec_copy_d2h(OSQPFloat *h_y, const cl_mem d_x, OSQPInt n);

/*
 * d_y[i] = h_x[i] for i in [0,n-1] (integers)
 */
void cl_vec_int_copy_h2d(cl_mem d_y, const OSQPInt *h_x, OSQPInt n);

/*
 * h_y[i] = d_x[i] for i in [0,n-1] (integers)
 */
void cl_vec_int_copy_d2h(OSQPInt *h_y, const cl_mem d_x, OSQPInt n);

/**
 * d_a[i] = sc for i in [0,n-1]
 */
void cl_vec_set_sc(cl_mem d_a, OSQPFloat sc, OSQPInt n);

/**
 *           | sc_if_neg   d_test[i]  < 0
 * d_a[i] = <  sc_if_zero  d_test[i] == 0   for i in [0,n-1]
 *           | sc_if_pos   d_test[i]  > 0
 */
void cl_vec_set_sc_cond(cl_mem d_a, const cl_mem d_test, OSQPFloat sc_if_neg,
                        OSQPFloat sc_if_zero, OSQPFloat sc_if_pos, OSQPInt n);

/**
 * d_a[i] *= sc for i in [0,n-1]
 */
void cl_vec_mult_sc(cl_mem d_a, OSQPFloat sc, OSQPInt n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] for i in [0,n-1]
 */
void cl_vec_add_scaled(cl_mem d_x, const cl_mem d_a, const cl_mem d_b,
                       OSQPFloat sca, OSQPFloat scb, OSQPInt n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] + scc * d_c[i] for i in [0,n-1]
 */
void cl_vec_add_scaled3(cl_mem d_x, const cl_mem d_a, const cl_mem d_b,
                        const cl_mem d_c, OSQPFloat sca, OSQPFloat scb,
                        OSQPFloat scc, OSQPInt n);

/**
 * h_res = |d_x|_inf
 */
void cl_vec_norm_inf(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res);

/**
 * res = |d_x|_2
 */
void cl_vec_norm_2(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res);

/**
 * h_res = |S*v|_inf
 */
void cl_vec_scaled_norm_inf(const cl_mem d_S, const cl_mem d_v, OSQPInt n,
                            OSQPFloat *h_res);

/**
 * h_res = |d_a - d_b|_inf
 */
void cl_vec_diff_norm_inf(const cl_mem d_a, const cl_mem d_b, OSQPInt n,
                          OSQPFloat *h_res);

/**
 * h_res = sum(|d_x|)
 */
void cl_vec_norm_1(const cl_mem d_x, OSQPInt n, OSQPFloat *h_res);

/**
 * h_res = d_a' * d_b
 */
void cl_vec_prod(const cl_mem d_a, const cl_mem d_b, OSQPInt n,
                 OSQPFloat *h_res);

/**
 *          | d_a' * max(d_b, 0)  sign ==  1
 * h_res = <  d_a' * min(d_b, 0)  sign == -1
 *          | d_a' * d_b          otherwise
 */
void cl_vec_prod_signed(const cl_mem d_a, const cl_mem d_b, OSQPInt sign,
                        OSQPInt n, OSQPFloat *h_res);

/**
 * d_c[i] = d_a[i] * d_b[i] for i in [0,n-1]
 */
void cl_vec_ew_prod(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n);

/**
 * h_res = all(a == b)
 */
void cl_vec_eq(const cl_mem a, const cl_mem b, OSQPFloat tol, OSQPInt n,
               OSQPInt *h_res);

/**
 * h_res = all(d_l <= d_u)
 */
void cl_vec_leq(const cl_mem d_l, const cl_mem d_u, OSQPInt n, OSQPInt *h_res);

/**
 * d_x[i] = min( max(d_z[i], d_l[i]), d_u[i] ) for i in [0,n-1]
 */
void cl_vec_bound(cl_mem d_x, const cl_mem d_z, const cl_mem d_l,
                  const cl_mem d_u, OSQPInt n);

/**
 *           | 0.0               d_l < -infval AND d_u > +infval
 * d_y[i] = <  min(d_y[i], 0.0)  d_u > +infval
 *           | max(d_y[i], 0.0)  d_l < -infval
 */
void cl_vec_project_polar_reccone(cl_mem d_y, const cl_mem d_l,
                                  const cl_mem d_u, OSQPFloat infval,
                                  OSQPInt n);

/**
 *          | d_y[i] \in [-tol,tol]  d_l[i] > -infval AND d_u[i] < +infval
 * h_res = <  d_y[i] < +tol          d_l[i] < -infval AND d_u[i] < +infval
 *          | d_y[i] > -tol          d_l[i] > -infval AND d_u[i] > +infval
 */
void cl_vec_in_reccone(const cl_mem d_y, const cl_mem d_l, const cl_mem d_u,
                       OSQPFloat infval, OSQPFloat tol, OSQPInt n,
                       OSQPInt *h_res);

/**
 * d_b[i] = 1 / d_a[i] for i in [0,n-1]
 */
void cl_vec_reciprocal(cl_mem d_b, const cl_mem d_a, OSQPInt n);

/**
 * d_a[i] = sqrt(d_a[i]) for i in [0,n-1]
 */
void cl_vec_sqrt(cl_mem d_a, OSQPInt n);

/**
 * d_c[i] = max(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cl_vec_max(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n);

/**
 * d_c[i] = min(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cl_vec_min(cl_mem d_c, const cl_mem d_a, const cl_mem d_b, OSQPInt n);

void cl_vec_bounds_type(cl_mem d_iseq, const cl_mem d_l, const cl_mem d_u,
                        OSQPFloat infval, OSQPFloat tol, OSQPInt n,
                        OSQPInt *h_has_changed);

void cl_vec_set_sc_if_lt(cl_mem d_x, const cl_mem d_z, OSQPFloat testval,
                         OSQPFloat newval, OSQPInt n);

void cl_vec_set_sc_if_gt(cl_mem d_x, const cl_mem d_z, OSQPFloat testval,
                         OSQPFloat newval, OSQPInt n);

void cl_vec_gather(OSQPInt nnz, const cl_mem d_y, cl_mem d_xVal,
                   const cl_mem d_xInd);

/*******************************************************************************
 *                           Matrix Functions                                  *
 *******************************************************************************/

/**
 * S = sc * S
 */
void cl_mat_mult_sc(csr *S, csr *At, OSQPFloat sc);

/**
 * S = D * S
 */
void cl_mat_lmult_diag(csr *S, csr *At, const cl_mem d_diag);

/**
 * S = S * D
 */
void cl_mat_rmult_diag(csr *S, csr *At, const cl_mem d_diag);

/**
 * X = S * D
 * X->val values are stored in d_buffer.
 */
void cl_mat_rmult_diag_new(const csr *S, cl_mem d_buffer, const cl_mem d_diag);

/**
 * d_y = alpha * A*d_x + beta*d_y
 */
void cl_mat_Axpy(const csr *A, const OSQPVectorf *vecx, OSQPVectorf *vecy,
                 OSQPFloat alpha, OSQPFloat beta);

/**
 * d_res[i] = |S_i|_inf where S_i is i-th row of S
 */
void cl_mat_row_norm_inf(const csr *S, cl_mem d_res);

#endif