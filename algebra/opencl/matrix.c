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

#include "algebra_types.h"
#include "lin_alg.h"
#include "opencl_csr.h"
#include "opencl_lin_alg.h"

#include "printing.h"
#include <assert.h>

/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

OSQPInt OSQPMatrix_is_eq(const OSQPMatrix *A, const OSQPMatrix *B,
                         OSQPFloat tol) {

  return cl_csr_is_eq(A->S, B->S, tol);
}

OSQPMatrix *OSQPMatrix_new_from_csc(const OSQPCscMatrix *M, OSQPInt is_triu) {

  OSQPMatrix *out = (OSQPMatrix *)c_calloc(1, sizeof(OSQPMatrix));

  if (!out) {
    return OSQP_NULL;
  }

  if (is_triu) {
    /* Initialize P */
    out->At = NULL; /* indicates a symmetric matrix */
    out->P_triu_nnz = M->p[M->n];

    /*  */
    cl_mat_init_P(M, &out->S, &out->cl_d_P_triu_val,
                  &out->cl_d_P_triu_to_full_ind, &out->cl_d_P_diag_ind);
  } else {
    /* Initialize A */
    cl_mat_init_A(M, &out->S, &out->At, &out->cl_d_A_to_At_ind);
  }

  return out;
}

void OSQPMatrix_update_values(OSQPMatrix *mat, const OSQPFloat *Mx_new,
                              const OSQPInt *Mx_new_idx, OSQPInt Mx_new_n) {

  if (mat->At) { /* not symmetric */
    cl_mat_update_A(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, &mat->At,
                    mat->cl_d_A_to_At_ind);
  } else {
    cl_mat_update_P(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, mat->d_P_triu_val,
                    mat->d_P_triu_to_full_ind, mat->d_P_diag_ind,
                    mat->P_triu_nnz);
  }
}

OSQPInt OSQPMatrix_get_m(const OSQPMatrix *mat) { return mat->S->m; }

OSQPInt OSQPMatrix_get_n(const OSQPMatrix *mat) { return mat->S->n; }

OSQPInt OSQPMatrix_get_nz(const OSQPMatrix *mat) {
  return mat->At ? mat->S->nnz : mat->P_triu_nnz;
}

void OSQPMatrix_mult_scalar(OSQPMatrix *mat, OSQPFloat sc) {

  cl_mat_mult_sc(mat->S, mat->At, sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix *mat, const OSQPVectorf *D) {

  cl_mat_lmult_diag(mat->S, mat->At, D->cl_vec);
}

void OSQPMatrix_rmult_diag(OSQPMatrix *mat, const OSQPVectorf *D) {

  cl_mat_rmult_diag(mat->S, mat->At, D->cl_vec);
}

void OSQPMatrix_Axpy(const OSQPMatrix *mat, const OSQPVectorf *x,
                     OSQPVectorf *y, OSQPFloat alpha, OSQPFloat beta) {

  if (mat->S->nnz == 0 || alpha == 0.0) {
    //   /*  y = beta * y  */
    cl_vec_mult_sc(y->cl_vec, beta, y->length);
    return;
  }
  //
  if ((x->length > 0) && (y->length > 0)) {
    cl_mat_Axpy(mat->S, x, y, alpha, beta);
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix *mat, const OSQPVectorf *x,
                      OSQPVectorf *y, OSQPFloat alpha, OSQPFloat beta) {

  if (mat->At->nnz == 0 || alpha == 0.0) {
    /*  y = beta * y  */
    cl_vec_mult_sc(y->cl_vec, beta, y->length);
    return;
  }

  if ((x->length > 0) && (y->length > 0)) {
    cl_mat_Axpy(mat->At, x, y, alpha, beta);
  }
}

void OSQPMatrix_col_norm_inf(const OSQPMatrix *mat, OSQPVectorf *res) {

  if (mat->At) {
    cl_mat_row_norm_inf(mat->At, res->cl_vec);
  } else {
    cl_mat_row_norm_inf(mat->S, res->cl_vec);
  }
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *mat, OSQPVectorf *res) {

  cl_mat_row_norm_inf(mat->S, res->cl_vec);
}

void OSQPMatrix_free(OSQPMatrix *mat) {
  if (mat) {
    cl_mat_free(mat->S);
    cl_mat_free(mat->At);

    //  cl_free((void **) &mat->d_A_to_At_ind);
    //  cl_free((void **) &mat->d_P_triu_to_full_ind);
    //  cl_free((void **) &mat->d_P_diag_ind);
    //  cl_free((void **) &mat->d_P_triu_val);
    c_free(mat);
  }
}

OSQPMatrix *OSQPMatrix_submatrix_byrows(const OSQPMatrix *mat,
                                        const OSQPVectori *rows) {

  OSQPMatrix *out;

  if (!mat->At) {
    c_eprint("row selection not implemented for partially filled matrices");
    return OSQP_NULL;
  }

  out = (OSQPMatrix *)c_calloc(1, sizeof(OSQPMatrix));
  if (!out) {
    return OSQP_NULL;
  }

  cl_submat_byrows(mat->S, rows->cl_vec, &out->S, &out->At);

  return out;
}

void OSQPMatrix_AtDA_extract_diag(const OSQPMatrix *A, const OSQPVectorf *D,
                                  OSQPVectorf *d) {
  OSQPInt j, i;
  const OSQPInt n = A->S->n;
  const OSQPInt *Ap = A->S->row_ptr;
  const OSQPInt *Ai = A->S->row_ind;
  const OSQPFloat *Ax = A->S->val;

  // OSQPVectorf_data(d);

  // Each entry of output vector is for a column, so cycle over columns
  for (j = 0; j < n; j++) {
    d->d_val[j] = 0;
    // Iterate over each entry in the column
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      d->d_val[j] += Ax[i] * Ax[i] * D->d_val[Ai[i]];
    }
  }
}

void OSQPMatrix_extract_diag(const OSQPMatrix *A, OSQPVectorf *d) {

  OSQPInt i, ptr;
  const OSQPInt n = A->S->n;
  const OSQPInt *Ap = A->S->row_ptr;
  const OSQPInt *Ai = A->S->row_ind;
  const OSQPFloat *Ax = A->S->val;

  /* Initialize output vector to 0 */
  cl_vec_set_sc(d->cl_vec, 0.0, n);

  /* Loop over columns to find when the row index equals column index */
  for (i = 0; i < n; i++) {
    for (ptr = Ap[i]; ptr < Ap[i + 1]; ptr++) {
      // assert(ptr < A->S->nnz);
      if (Ai[ptr] == i) {
        d->d_val[i] = Ax[ptr];
      }
    }
  }
}
