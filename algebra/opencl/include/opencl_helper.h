#pragma once
#include <CL/cl.h>
#include <stdio.h>

#ifdef __cplusplus /*	C++ Environment	*/
extern "C" {
#endif

typedef struct device_information_t {
  cl_bool unifiedMemory;

  cl_uint alignedSize;
  cl_ulong maxMemSize;
  cl_uint workDim;
  size_t work_size[3];

  size_t localworksize[1];
  cl_ulong preferedWorkSize[3];

} DeviceInformation;

typedef struct opencl_handle_t {

  cl_context context;
  unsigned int nDevices;
  cl_device_id *devices;
  cl_mem buf;
  cl_program program;

  DeviceInformation *deviceInformations;

  cl_command_queue queue;

  /*  */
  cl_kernel vec_norm_inf_kernel;

  cl_kernel vec_norm_1_kernel; //
  cl_kernel vec_set_sc_kernel; //
  cl_kernel vec_mult_sc;       //

  cl_kernel vec_add_scaled_kernel;
  cl_kernel vec_add_scaled3_kernel;

  cl_kernel vec_set_sc_cond_kernel;           //
  cl_kernel vec_prod_pos_kernel;              //
  cl_kernel vec_prod_neg_kernel;              //
  cl_kernel vec_prod_kernel;                  //
  cl_kernel vec_ew_prod_kernel;               //
  cl_kernel vec_eq_kernel;                    //
  cl_kernel vec_leq_kernel;                   //
  cl_kernel vec_bound_kernel;                 //
  cl_kernel vec_project_polar_reccone_kernel; //
  cl_kernel vec_in_reccone_kernel;            //
  cl_kernel vec_reciprocal_kernel;            //
  cl_kernel vec_sqrt_kernel;                  //
  cl_kernel vec_max_kernel;                   //
  cl_kernel vec_min_kernel;                   //
  cl_kernel vec_bounds_type_kernel;           //
  cl_kernel vec_set_sc_if_lt_kernel;          //
  cl_kernel vec_set_sc_if_gt_kernel;          //
  cl_kernel mat_lmult_diag_kernel;            //
  cl_kernel mat_rmult_diag_kernel;            //
  cl_kernel mat_rmult_diag_new_kernel;        //
  cl_kernel vec_abs_kernel;                   //
  cl_kernel scatter_kernel;                   //
  cl_kernel gather_kernel;                    //
  cl_kernel abs_kernel;                       //

  cl_kernel vec_gather_kernel;

  /*  matrix  */
  cl_kernel fill_full_matrix_kernel;
  cl_kernel add_diagonal_kernel;
  cl_kernel reduce_permutation_kernel;
  cl_kernel get_diagonal_indices_kernel;
  cl_kernel predicate_generator_kernel;
  cl_kernel compact_rows;
  cl_kernel compact;
  cl_kernel vector_init_abs_kernel;
  cl_kernel csr_eq_kernel;
  cl_kernel Axpy_mat_kernel;
  cl_kernel inclusive_scan_kernel;
  cl_kernel offsets_to_indices_kernel;

  cl_mem scratch;

} OpenCL_Handle_t;

/**
 * Get OpenCL error string.
 * @return non-null context.
 */
extern const char *getCLStringError(unsigned int errorcode);

extern cl_context createCLContext(cl_uint *ndevices, cl_device_id **devices,
                                  cl_uint selected_platform);

extern cl_program createProgramSource(cl_context context, unsigned int nDevices,
                                      cl_device_id *devices, int numSources,
                                      const char **cfilename);

extern cl_program createProgramBinary(cl_context context, unsigned int nDevices,
                                      cl_device_id *devices,
                                      const char *cfilename);

extern cl_kernel createKernel(cl_program program, const char *name);

extern cl_command_queue createCommandQueue(cl_context context,
                                           cl_device_id device);

extern unsigned int align(const unsigned int size,
                          const unsigned int alignment);

#ifdef __cplusplus /*	C++ Environment	*/
}
#endif