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
#include "algebra_vector.h"
#include "include/opencl_helper.h"
#include "include/opencl_lin_alg.h"
#include "lin_alg.h"
#include "opencl_alloc.h"
#include "osqp_api_types.h"
#include <CL/cl.h>

extern OpenCL_Handle_t *handle;

/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

OSQPInt OSQPVectorf_is_eq(const OSQPVectorf *a, const OSQPVectorf *b,
                          OSQPFloat tol) {

  OSQPInt res = 0;

  if (a->length != b->length) {
    return 0;
  }

  cl_vec_eq(a->cl_vec, b->cl_vec, tol, a->length, &res);

  return res;
}

OSQPVectorf *OSQPVectorf_new(const OSQPFloat *a, OSQPInt length) {

  OSQPVectorf *out = OSQPVectorf_malloc(length);
  if (!out) {
    return OSQP_NULL;
  }

  if (length > 0) {
    OSQPVectorf_from_raw(out, a);
  }

  return out;
}

OSQPVectorf *OSQPVectorf_malloc(OSQPInt length) {

  OSQPVectorf *b = (OSQPVectorf *)c_malloc(sizeof(OSQPVectorf));
  if (!b) {
    return OSQP_NULL;
  }

  b->length = length;
  if (length) {
    cl_vecf_create(&b->cl_vec, NULL, length);
    if (!(b->cl_vec)) {
      c_free(b);
      b = OSQP_NULL;
    }
    /*  Map data. */
    cl_buffer_region region = {0, length * sizeof(OSQPFloat)};
    b->d_val = cl_host_map(b->cl_vec, CL_MAP_READ | CL_MAP_WRITE, &region);

    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  } else {
    b->d_val = OSQP_NULL;
    b->cl_vec = NULL;
  }
  return b;
}

OSQPVectorf *OSQPVectorf_calloc(OSQPInt length) {

  OSQPVectorf *b = (OSQPVectorf *)c_malloc(sizeof(OSQPVectorf));
  if (!b) {
    return OSQP_NULL;
  }

  b->length = length;
  if (length) {

    cl_vecf_create(&b->cl_vec, NULL, length);
    if (!(b->cl_vec)) {
      c_free(b);
      b = OSQP_NULL;
    }

    const cl_buffer_region region = {0, length * sizeof(OSQPFloat)};
    cl_memset(b->cl_vec, 0, 0, region.size);

    /*  Map data. */
    b->d_val = cl_host_map(b->cl_vec, CL_MAP_READ | CL_MAP_WRITE, &region);

    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  } else {
    b->d_val = OSQP_NULL;
    b->cl_vec = NULL;
  }
  return b;
}

OSQPVectori *OSQPVectori_new(const OSQPInt *a, OSQPInt length) {

  OSQPVectori *out = OSQPVectori_malloc(length);
  if (!out) {
    return OSQP_NULL;
  }

  if (length > 0) {
    OSQPVectori_from_raw(out, a);
  }

  return out;
}

OSQPVectori *OSQPVectori_malloc(OSQPInt length) {

  OSQPVectori *b = (OSQPVectori *)c_malloc(sizeof(OSQPVectori));
  if (!b) {
    return OSQP_NULL;
  }

  b->length = length;
  if (length) {

    /*  */
    cl_veci_create(&b->cl_vec, NULL, length);
    if (!(b->cl_vec)) {
      c_free(b);
      b = OSQP_NULL;
    }

    /*  Map data. */
    const cl_buffer_region region = {0, length * sizeof(OSQPFloat)};
    b->d_val = cl_host_map(b->cl_vec, CL_MAP_READ, &region);

    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  } else {
    b->d_val = OSQP_NULL;
    b->cl_vec = NULL;
  }
  return b;
}

OSQPVectori *OSQPVectori_calloc(OSQPInt length) {

  OSQPVectori *b = (OSQPVectori *)c_malloc(sizeof(OSQPVectori));
  if (!b) {
    return OSQP_NULL;
  }

  b->length = length;

  if (length) {
    cl_veci_create(&b->cl_vec, NULL, length);
    if (!(b->cl_vec)) {
      c_free(b);
      b = OSQP_NULL;
    }

    const cl_buffer_region region = {0, length * sizeof(OSQPInt)};
    cl_memset(b->cl_vec, 0, 0, region.size);

    /*  Map data. */
    b->d_val = cl_host_map(b->cl_vec, CL_MAP_READ, &region);

    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  } else {
    b->d_val = OSQP_NULL;
    b->cl_vec = NULL;
  }
  return b;
}

OSQPVectorf *OSQPVectorf_copy_new(const OSQPVectorf *a) {

  OSQPVectorf *b = OSQPVectorf_malloc(a->length);

  if (b) {
    cl_memcpy_d2d(b->cl_vec, a->cl_vec, a->length * sizeof(OSQPFloat));
  }

  return b;
}

void OSQPVectorf_free(OSQPVectorf *a) {

  if (a && a->cl_vec && a->d_val) {
    cl_host_ummap(a->cl_vec, a->d_val);
    cl_vec_destroy(a->cl_vec);
  }
  c_free(a);
}

void OSQPVectori_free(OSQPVectori *a) {

  if (a && a->cl_vec && a->d_val) {
    cl_host_ummap(a->cl_vec, a->d_val);
    cl_vec_destroy(a->cl_vec);
  }
  c_free(a);
}

OSQPVectorf *OSQPVectorf_view(const OSQPVectorf *a, OSQPInt head,
                              OSQPInt length) {

  OSQPVectorf *view = (OSQPVectorf *)c_malloc(sizeof(OSQPVectorf));
  if (view) {
    view->length = length;
    view->d_val = a->d_val + head;

    const cl_buffer_region region = {head, length};
    cl_allocate_sub_mem(a->cl_vec, CL_MEM_READ_WRITE, 1, &region,
                        &view->cl_vec);
  }
  return view;
}

void OSQPVectorf_view_free(OSQPVectorf *a) { c_free(a); }

OSQPInt OSQPVectorf_length(const OSQPVectorf *a) { return a->length; }
OSQPInt OSQPVectori_length(const OSQPVectori *a) { return a->length; }

void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a) {

  if (a) {
    cl_memcpy_d2d(b->cl_vec, a->cl_vec, a->length * sizeof(OSQPFloat));
  }
}

void OSQPVectorf_from_raw(OSQPVectorf *b, const OSQPFloat *av) {

  if (av) {
    cl_vec_copy_h2d(b->cl_vec, av, b->length);
  }
}

void OSQPVectori_from_raw(OSQPVectori *b, const OSQPInt *av) {
  cl_vec_int_copy_h2d(b->cl_vec, av, b->length);
}

void OSQPVectorf_to_raw(OSQPFloat *bv, const OSQPVectorf *a) {

  cl_vec_copy_d2h(bv, a->d_val, a->length);
}

void OSQPVectori_to_raw(OSQPInt *bv, const OSQPVectori *a) {

  cl_memcpy_d2h(bv, a->cl_vec, a->length * sizeof(OSQPInt));
}

void OSQPVectorf_set_scalar(OSQPVectorf *a, OSQPFloat sc) {

  cl_vec_set_sc(a->cl_vec, sc, a->length);
}

void OSQPVectorf_set_scalar_conditional(OSQPVectorf *a, const OSQPVectori *test,
                                        OSQPFloat sc_if_neg,
                                        OSQPFloat sc_if_zero,
                                        OSQPFloat sc_if_pos) {

  cl_vec_set_sc_cond(a->cl_vec, test->cl_vec, sc_if_neg, sc_if_zero, sc_if_pos,
                     a->length);
}

void OSQPVectorf_mult_scalar(OSQPVectorf *a, OSQPFloat sc) {

  if (sc == 1.0 || !a->d_val) {
    return;
  }
  cl_vec_mult_sc(a->cl_vec, sc, a->length);
}

void OSQPVectorf_plus(OSQPVectorf *x, const OSQPVectorf *a,
                      const OSQPVectorf *b) {

  cl_vec_add_scaled(x->cl_vec, a->cl_vec, b->cl_vec, 1.0, 1.0, a->length);
}

void OSQPVectorf_minus(OSQPVectorf *x, const OSQPVectorf *a,
                       const OSQPVectorf *b) {

  cl_vec_add_scaled(x->cl_vec, a->cl_vec, b->cl_vec, 1.0, -1.0, a->length);
}

void OSQPVectorf_add_scaled(OSQPVectorf *x, OSQPFloat sca, const OSQPVectorf *a,
                            OSQPFloat scb, const OSQPVectorf *b) {

  cl_vec_add_scaled(x->cl_vec, a->cl_vec, b->cl_vec, sca, scb, x->length);
}

void OSQPVectorf_add_scaled3(OSQPVectorf *x, OSQPFloat sca,
                             const OSQPVectorf *a, OSQPFloat scb,
                             const OSQPVectorf *b, OSQPFloat scc,
                             const OSQPVectorf *c) {

  cl_vec_add_scaled3(x->cl_vec, a->cl_vec, b->cl_vec, c->cl_vec, sca, scb, scc,
                     x->length);
}

OSQPFloat OSQPVectorf_norm_inf(const OSQPVectorf *v) {

  OSQPFloat normval;

  if (v->length) {
    cl_vec_norm_inf(v->cl_vec, v->length, &normval);
  } else {
    normval = 0.0;
  }

  return normval;
}

OSQPFloat OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S,
                                      const OSQPVectorf *v) {

  OSQPFloat normval;

  if (v->length) {
    cl_vec_scaled_norm_inf(S->cl_vec, v->cl_vec, v->length, &normval);
  } else {
    normval = 0.0;
  }

  return normval;
}

OSQPFloat OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                    const OSQPVectorf *b) {

  OSQPFloat normDiff;

  if (a->length) {
    cl_vec_diff_norm_inf(a->cl_vec, b->cl_vec, a->length, &normDiff);
  } else {
    normDiff = 0.0;
  }

  return normDiff;
}

OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf *a) {

  OSQPFloat val = 0.0;

  if (a->length) {
    cl_vec_norm_1(a->cl_vec, a->length, &val);
  }

  return val;
}

OSQPFloat OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b) {

  OSQPFloat dotprod;

  if (a->length) {
    cl_vec_prod(a->cl_vec, b->cl_vec, a->length, &dotprod);
  } else {
    dotprod = 0.0;
  }

  return dotprod;
}

OSQPFloat OSQPVectorf_dot_prod_signed(const OSQPVectorf *a,
                                      const OSQPVectorf *b, OSQPInt sign) {

  OSQPFloat dotprod;

  if (a->length) {
    cl_vec_prod_signed(a->cl_vec, b->cl_vec, sign, a->length, &dotprod);
  } else {
    dotprod = 0.0;
  }

  return dotprod;
}

void OSQPVectorf_ew_prod(OSQPVectorf *c, const OSQPVectorf *a,
                         const OSQPVectorf *b) {

  if (c->length) {
    cl_vec_ew_prod(c->cl_vec, a->cl_vec, b->cl_vec, c->length);
  }
}

OSQPInt OSQPVectorf_all_leq(const OSQPVectorf *l, const OSQPVectorf *u) {

  OSQPInt res;

  cl_vec_leq(l->cl_vec, u->cl_vec, l->length, &res);

  return res;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf *x, const OSQPVectorf *z,
                              const OSQPVectorf *l, const OSQPVectorf *u) {

  cl_vec_bound(x->cl_vec, z->cl_vec, l->cl_vec, u->cl_vec, x->length);
}

void OSQPVectorf_project_polar_reccone(OSQPVectorf *y, const OSQPVectorf *l,
                                       const OSQPVectorf *u, OSQPFloat infval) {

  cl_vec_project_polar_reccone(y->cl_vec, l->cl_vec, u->cl_vec, infval,
                               y->length);
}

OSQPInt OSQPVectorf_in_reccone(const OSQPVectorf *y, const OSQPVectorf *l,
                               const OSQPVectorf *u, OSQPFloat infval,
                               OSQPFloat tol) {

  OSQPInt res;

  cl_vec_in_reccone(y->cl_vec, l->cl_vec, u->cl_vec, infval, tol, y->length,
                    &res);
  return res;
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf *b, const OSQPVectorf *a) {

  if (b->length) {
    cl_vec_reciprocal(b->cl_vec, a->cl_vec, b->length);
  }
}

void OSQPVectorf_ew_sqrt(OSQPVectorf *a) {

  if (a->length) {
    cl_vec_sqrt(a->cl_vec, a->length);
  }
}

void OSQPVectorf_ew_max_vec(OSQPVectorf *c, const OSQPVectorf *a,
                            const OSQPVectorf *b) {

  if (c->length) {
    cl_vec_max(c->cl_vec, a->cl_vec, b->cl_vec, c->length);
  }
}

void OSQPVectorf_ew_min_vec(OSQPVectorf *c, const OSQPVectorf *a,
                            const OSQPVectorf *b) {

  if (c->length) {
    cl_vec_min(c->cl_vec, a->cl_vec, b->cl_vec, c->length);
  }
}

OSQPInt OSQPVectorf_ew_bounds_type(OSQPVectori *iseq, const OSQPVectorf *l,
                                   const OSQPVectorf *u, OSQPFloat tol,
                                   OSQPFloat infval) {

  OSQPInt has_changed;

  cl_vec_bounds_type(iseq->cl_vec, l->cl_vec, u->cl_vec, infval, tol,
                     iseq->length, &has_changed);

  return has_changed;
}

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf *x, const OSQPVectorf *z,
                                  OSQPFloat testval, OSQPFloat newval) {

  cl_vec_set_sc_if_lt(x->cl_vec, z->cl_vec, testval, newval, x->length);
}

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf *x, const OSQPVectorf *z,
                                  OSQPFloat testval, OSQPFloat newval) {

  cl_vec_set_sc_if_gt(x->cl_vec, z->cl_vec, testval, newval, x->length);
}
