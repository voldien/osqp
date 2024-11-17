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

#ifndef OPENCL_CSR_H
#define OPENCL_CSR_H

#include "algebra_types.h"
#include "osqp_api_types.h"

void cl_mat_init_P(const OSQPCscMatrix *mat, csr **P, cl_mem *d_P_triu_val,
                   cl_mem *d_P_triu_to_full_ind, cl_mem *d_P_diag_ind);

void cl_mat_init_A(const OSQPCscMatrix *mat, csr **A, csr **At,
                   cl_mem *d_A_to_At_ind);

void cl_mat_update_P(const OSQPFloat *Px, const OSQPInt *Px_idx, OSQPInt Px_n,
                     csr **P, OSQPFloat *d_P_triu_val,
                     OSQPInt *d_P_triu_to_full_ind, OSQPInt *d_P_diag_ind,
                     OSQPInt P_triu_nnz);

void cl_mat_update_A(const OSQPFloat *Ax, const OSQPInt *Ax_idx, OSQPInt Ax_n,
                     csr **A, csr **At, cl_mem d_A_to_At_ind);

void cl_mat_free(csr *mat);

OSQPInt cl_csr_is_eq(const csr *A, const csr *B, OSQPFloat tol);

void cl_submat_byrows(const csr *A, const cl_mem d_rows, csr **Ared,
                      csr **Aredt);

void inclusive_scan(const cl_mem A, cl_mem result, OSQPInt n);

#endif /* ifndef CUDA_CSR_H */
