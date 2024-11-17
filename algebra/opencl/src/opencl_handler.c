#include "opencl_handler.h"
#include "opencl_helper.h"
#include "osqp_api_types.h"
#include <CL/cl.h>

OpenCL_Handle_t *handle = NULL;

OSQPInt opencl_init(OSQPInt device) {

  handle = calloc(1, sizeof(OpenCL_Handle_t));
  /*  Setup opencl  */
  if (!handle) {
    opencl_release();
    return 1;
  }
  device = 0;

  /*  Create OpenCL context.  */
  handle->context =
      createCLContext(&handle->nDevices, &handle->devices, device);
  if (!handle->context) {
    opencl_release();
    return 1;
  }

  handle->deviceInformations =
      calloc(handle->nDevices, sizeof(DeviceInformation));

  for (int i = 0; i < handle->nDevices; i++) {
    cl_int err;
    /*  Work group - Global. */
    err =
        clGetDeviceInfo(handle->devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(handle->deviceInformations[i].workDim),
                        &handle->deviceInformations[i].workDim, NULL);

    err = clGetDeviceInfo(handle->devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(handle->deviceInformations[i].localworksize),
                          &handle->deviceInformations[i].localworksize, NULL);

    /*  Work Item - Local   */
    err = clGetDeviceInfo(handle->devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(handle->deviceInformations[i].work_size),
                          &handle->deviceInformations[i].work_size, NULL);

    /*  Aligned data size.  */
    err =
        clGetDeviceInfo(handle->devices[i], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                        sizeof(handle->deviceInformations[i].alignedSize),
                        &handle->deviceInformations[i].alignedSize, NULL);

    /*  Aligned data size.  */
    err = clGetDeviceInfo(handle->devices[i], CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                          sizeof(handle->deviceInformations[i].alignedSize),
                          &handle->deviceInformations[i].alignedSize, NULL);

    /*  Max allocation size.  */
    err = clGetDeviceInfo(handle->devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY,
                          sizeof(handle->deviceInformations[i].unifiedMemory),
                          &handle->deviceInformations[i].unifiedMemory, NULL);

    /*  Max allocation size.  */
    err = clGetDeviceInfo(handle->devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(handle->deviceInformations[i].maxMemSize),
                          &handle->deviceInformations[i].maxMemSize, NULL);
  }

#define ELEMENTS_PER_THREAD (8)
#define THREADS_PER_BLOCK (1024)
#define NUMBER_OF_BLOCKS (2)
#define NUMBER_OF_SM (68)

  /*  Create OpenCL command queue.    */
  handle->queue = createCommandQueue(handle->context, handle->devices[0]);
  if (!handle->queue) {
    opencl_release();
    return 1;
  }

  /*  Load program and kernel.    */
  const char *sources[] = {
      "/mnt/mjpdev/404/osqp/algebra/opencl/kernel/lin_algebra.cl",
      "/mnt/mjpdev/404/osqp/algebra/opencl/kernel/csr_matrix.cl"};
  handle->program = createProgramSource(handle->context, handle->nDevices,
                                        handle->devices, 2, sources);
  if (!handle->program) {
    opencl_release();
    return 1;
  }
  cl_uint num_kernels;
  cl_int err = clCreateKernelsInProgram(handle->program, 0, NULL, &num_kernels);

  {

    handle->vec_norm_inf_kernel =
        clCreateKernel(handle->program, "vec_norm_inf_kernel", &err);

    handle->vec_norm_1_kernel =
        clCreateKernel(handle->program, "vec_norm_1_kernel", &err);

    handle->vec_set_sc_kernel =
        clCreateKernel(handle->program, "vec_set_sc_kernel", &err);

    handle->vec_mult_sc =
        clCreateKernel(handle->program, "vec_mult_sc_kernel", &err);

    handle->vec_add_scaled_kernel =
        clCreateKernel(handle->program, "vec_add_scaled_kernel", &err);

    handle->vec_add_scaled3_kernel =
        clCreateKernel(handle->program, "vec_add_scaled3_kernel", &err);

    handle->vec_set_sc_cond_kernel =
        clCreateKernel(handle->program, "vec_set_sc_cond_kernel", &err);

    handle->vec_prod_kernel =
        clCreateKernel(handle->program, "vec_prod_kernel", &err);

    handle->vec_prod_pos_kernel =
        clCreateKernel(handle->program, "vec_prod_pos_kernel", &err);

    handle->vec_prod_neg_kernel =
        clCreateKernel(handle->program, "vec_prod_neg_kernel", &err);

    handle->vec_ew_prod_kernel =
        clCreateKernel(handle->program, "vec_ew_prod_kernel", &err);

    handle->vec_eq_kernel =
        clCreateKernel(handle->program, "vec_eq_kernel", &err);

    handle->vec_leq_kernel =
        clCreateKernel(handle->program, "vec_leq_kernel", &err);

    handle->vec_bound_kernel =
        clCreateKernel(handle->program, "vec_bound_kernel", &err);

    handle->vec_project_polar_reccone_kernel = clCreateKernel(
        handle->program, "vec_project_polar_reccone_kernel", &err);

    handle->vec_in_reccone_kernel =
        clCreateKernel(handle->program, "vec_in_reccone_kernel", &err);

    handle->vec_reciprocal_kernel =
        clCreateKernel(handle->program, "vec_reciprocal_kernel", &err);

    handle->vec_sqrt_kernel =
        clCreateKernel(handle->program, "vec_sqrt_kernel", &err);

    handle->vec_max_kernel =
        clCreateKernel(handle->program, "vec_max_kernel", &err);

    handle->vec_min_kernel =
        clCreateKernel(handle->program, "vec_min_kernel", &err);

    handle->vec_bounds_type_kernel =
        clCreateKernel(handle->program, "vec_bounds_type_kernel", &err);

    handle->vec_set_sc_if_lt_kernel =
        clCreateKernel(handle->program, "vec_set_sc_if_lt_kernel", &err);

    handle->vec_set_sc_if_gt_kernel =
        clCreateKernel(handle->program, "vec_set_sc_if_gt_kernel", &err);

    handle->mat_lmult_diag_kernel =
        clCreateKernel(handle->program, "mat_lmult_diag_kernel", &err);

    handle->mat_rmult_diag_kernel =
        clCreateKernel(handle->program, "mat_rmult_diag_kernel", &err);

    handle->mat_rmult_diag_new_kernel =
        clCreateKernel(handle->program, "mat_rmult_diag_new_kernel", &err);

    handle->vec_abs_kernel =
        clCreateKernel(handle->program, "vec_abs_kernel", &err);

    handle->scatter_kernel =
        clCreateKernel(handle->program, "scatter_kernel", &err);

    handle->gather_kernel =
        clCreateKernel(handle->program, "gather_kernel", &err);

    handle->abs_kernel = clCreateKernel(handle->program, "abs_kernel", &err);

    /*  matrix  */
    handle->fill_full_matrix_kernel =
        clCreateKernel(handle->program, "fill_full_matrix_kernel", &err);
    handle->add_diagonal_kernel =
        clCreateKernel(handle->program, "add_diagonal_kernel", &err);
    handle->reduce_permutation_kernel =
        clCreateKernel(handle->program, "reduce_permutation_kernel", &err);
    handle->get_diagonal_indices_kernel =
        clCreateKernel(handle->program, "get_diagonal_indices_kernel", &err);
    handle->predicate_generator_kernel =
        clCreateKernel(handle->program, "predicate_generator_kernel", &err);
    handle->compact = clCreateKernel(handle->program, "compact", &err);
    handle->compact_rows =
        clCreateKernel(handle->program, "compact_rows", &err);
    handle->vector_init_abs_kernel =
        clCreateKernel(handle->program, "vector_init_abs_kernel", &err);
    handle->csr_eq_kernel =
        clCreateKernel(handle->program, "csr_eq_kernel", &err);
    handle->Axpy_mat_kernel =
        clCreateKernel(handle->program, "Axpy_mat_kernel", &err);
    handle->inclusive_scan_kernel =
        clCreateKernel(handle->program, "inclusive_scan_kernel", &err);
    handle->offsets_to_indices_kernel =
        clCreateKernel(handle->program, "offsets_to_indices_kernel", &err);
  }

  return 0;
}

void opencl_release(void) {

  clReleaseKernel(handle->abs_kernel);

  clReleaseProgram(handle->program);
  clFlush(handle->queue);
  clReleaseCommandQueue(handle->queue);
  clReleaseContext(handle->context);

  if (handle) {
    free(handle->deviceInformations);
    free(handle);
  }
}