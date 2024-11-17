
#include "include/opencl_helper.h"
#include "lin_alg.h"
#include "lin_sys/direct/opencl_pardiso_interface.h"
#include "lin_sys/indirect/opencl_pcg_interface.h"
#include "opencl_handler.h"
#include "osqp_api_constants.h"
#include "osqp_api_types.h"
#include <CL/cl.h>

extern OpenCL_Handle_t *handle;

OSQPInt osqp_algebra_linsys_supported(void) {
  /* Only has QDLDL (direct solver) */
  return OSQP_CAPABILITY_INDIRECT_SOLVER | OSQP_CAPABILITY_DIRECT_SOLVER;
}

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void) {
  /* Prefer QDLDL (it is also the only one available) */
  return OSQP_DIRECT_SOLVER;
}

OSQPInt osqp_algebra_init_libs(OSQPInt device) {

  if (handle) {
    return 0;
  }
  return opencl_init(device);
}

void osqp_algebra_free_libs(void) { opencl_release(); }

OSQPInt osqp_algebra_name(char *name, OSQPInt nameLen) {

  // size_t valueSize = 0;
  // clGetDeviceInfo(handle->devices[0], CL_DEVICE_VERSION, 0, NULL,
  // &valueSize); value = (char *)malloc(valueSize);
  // clGetDeviceInfo(handle->devices[0], CL_DEVICE_VERSION, valueSize, value,
  // NULL);

  int runtimeVersion = 120;

  return snprintf(name, nameLen, "OpenCL %d.%d", runtimeVersion / 1000,
                  (runtimeVersion % 100) / 10);
}

OSQPInt osqp_algebra_device_name(char *name, OSQPInt nameLen) {
  /* No device name for built-in algebra */
  size_t valueSize = 0;
  clGetDeviceInfo(handle->devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
  clGetDeviceInfo(handle->devices[0], CL_DEVICE_NAME, valueSize, name, NULL);

  name[valueSize] = 0;
  return valueSize;
}

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
OSQPInt osqp_algebra_init_linsys_solver(
    LinSysSolver **s, const OSQPMatrix *P, const OSQPMatrix *A,
    const OSQPVectorf *rho_vec, const OSQPSettings *settings,
    OSQPFloat *scaled_prim_res, OSQPFloat *scaled_dual_res, OSQPInt polishing) {

  switch (settings->linsys_solver) {

  default:
  case OSQP_DIRECT_SOLVER:
    return init_linsys_solver_pardiso((pardiso_solver **)s, P, A, rho_vec,
                                      settings, polishing);
  case OSQP_INDIRECT_SOLVER:
    return init_linsys_solver_clpcg((clpcg_solver **)s, P, A, rho_vec, settings,
                                    scaled_prim_res, scaled_dual_res,
                                    polishing);
    break;
  }
  return 0;
}
