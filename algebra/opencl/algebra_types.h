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

#ifndef ALGEBRA_TYPES_H
#define ALGEBRA_TYPES_H

#include "osqp_api_types.h"
#include <CL/cl.h>
#include <stddef.h>

/* CSR matrix structure */
struct csr_t {
  OSQPInt m;   ///< number of rows
  OSQPInt n;   ///< number of columns
  OSQPInt nnz; ///< number of non-zero entries

  size_t SpMatBufferSize;
  cl_mem mainbuffer;

  OSQPFloat *val;   ///< numerical values (size nnz)
  OSQPInt *row_ptr; ///< row pointers (size m+1)
  OSQPInt *col_ind; ///< column indices (size nnz)
  OSQPInt *row_ind; ///< uncompressed row indices (size nnz), NULL if not needed

  cl_mem cl_val;
  cl_mem cl_row_ptr;
  cl_mem cl_col_ind;
  cl_mem cl_row_ind;
};

/*********************************************
 *   Internal definition of OSQPVector types
 *   and supporting definitions
 *********************************************/

struct OSQPVectori_ {
  OSQPInt *d_val;
  OSQPInt length;
  cl_mem cl_vec;
};

struct OSQPVectorf_ {
  OSQPFloat *d_val;
  OSQPInt length;
  cl_mem cl_vec;
};

/*********************************************
 *   Internal definition of OSQPMatrix type
 *   and supporting definitions
 *********************************************/

/* Matrix in CSR format stored in GPU memory */
typedef struct csr_t csr;

struct OSQPMatrix_ {
  csr *S;  /* P or A */
  csr *At; /* NULL if symmetric */

  OSQPInt P_triu_nnz;

  OSQPInt *d_A_to_At_ind;
  OSQPFloat *d_P_triu_val;
  OSQPInt *d_P_triu_to_full_ind;
  OSQPInt *d_P_diag_ind;

  /*  */
  cl_mem cl_d_A_to_At_ind;
  cl_mem cl_d_P_triu_val;
  cl_mem cl_d_P_triu_to_full_ind;
  cl_mem cl_d_P_diag_ind;
};

#endif /* ifndef ALGEBRA_TYPES_H */
