#define CL_VERSION_2_1
#include "opencl_helper.h"
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>

const char *getCLStringError(unsigned int errorcode) { return ""; }

cl_context createCLContext(cl_uint *ndevices, cl_device_id **devices,
                           cl_uint selected_platform) {

  cl_int err_num;
  cl_context context;
  cl_platform_id *platforms;

  cl_device_id gpuShareId;

  /*  Check if argument is non null reference.	*/
  assert(ndevices && devices);

  /*  Context properties.	*/
  cl_context_properties props[] = {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)NULL,
      NULL,
  };

  cl_uint nPlatforms;

  /*	Get Number of platform.	*/
  err_num = clGetPlatformIDs(0, NULL, &nPlatforms);
  if (err_num != CL_SUCCESS) {
    fprintf(stderr, "failed to get number of OpenCL platforms - %s\n",
            getCLStringError(err_num));
    return NULL;
  }

  /*  Get platforms.	*/
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * nPlatforms);
  err_num = clGetPlatformIDs(nPlatforms, platforms, NULL);
  if (err_num != CL_SUCCESS) {
    fprintf(stderr, "failed to get OpenCL platforms - %s\n",
            getCLStringError(err_num));
    return NULL;
  }

  unsigned int device_offset;
  unsigned int total_device_count = 0;
  if (selected_platform == -1) {

    /*	Iterate through each platform in till the
     *	platform associated with OpenGL context is found. */
    for (unsigned int x = 0; x < nPlatforms; x++) {
      size_t pvar = 4;
      char *extenions;

      /* TODO change to the device extension instead!    */

      // ciErrNum = clGetPlatformInfo(platforms[x], CL_PLATFORM_EXTENSIONS,
      //                              sizeof(extenions), &extenions, &pvar);
      // if (ciErrNum != CL_SUCCESS) {
      //   fprintf(stderr, "clGetPlatformInfo failed: %s\n",
      //           getCLStringError(ciErrNum));
      //   return NULL;
      // }

      /*  Done.   */
      break;
    }

    /*  Extract all GPU devices.  */
    for (cl_uint x = 0; x < nPlatforms; x++) {
      cl_uint nr_devices;
      err_num = clGetDeviceIDs(platforms[x], CL_DEVICE_TYPE_GPU, 0, NULL,
                               &nr_devices);
      if (err_num != CL_SUCCESS) {
        fprintf(stderr, "failed to get Device ID number - %s\n",
                getCLStringError(err_num));
        return NULL;
      }
      total_device_count += nr_devices;

      err_num = clGetDeviceIDs(platforms[x], CL_DEVICE_TYPE_CPU, 0, NULL,
                               &nr_devices);
      if (err_num != CL_SUCCESS) {
        fprintf(stderr, "failed to get Device ID number - %s\n",
                getCLStringError(err_num));
        return NULL;
      }
      total_device_count += nr_devices;
    }
    for (cl_uint x = 0; x < nPlatforms; x++) {
    }
  }

  {
    // CL_DEVICE_COMPILER_AVAILABLE
    //  clGetDeviceInfo
    /*	get device ids for the GPUS.	*/
    err_num = clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_GPU,
                             0, NULL, ndevices);
    if (err_num != CL_SUCCESS) {
      fprintf(stderr, "failed to get Device ID number - %s\n",
              getCLStringError(err_num));
      return NULL;
    }

    *devices = malloc(sizeof(cl_device_id) * total_device_count);
    err_num = clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_GPU,
                             *ndevices, *devices, ndevices);
    if (err_num != CL_SUCCESS) {
      fprintf(stderr, "failed to get Device ID poiners - %s\n",
              getCLStringError(err_num));
      return NULL;
    }
  }

  props[1] = (cl_context_properties)platforms[selected_platform];

  /*	Create context.	*/
  assert(props[1] && props[0]);
  context = clCreateContext(props, *ndevices, *devices, NULL, NULL, &err_num);

  /*  Error check.    */
  if (context == NULL || err_num != CL_SUCCESS) {
    fprintf(stderr, "failed to create OpenCL context - %s\n",
            getCLStringError(err_num));
    return NULL;
  }

  free(platforms);

  return context;
}

cl_program createProgramSource(cl_context context, unsigned int nDevices,
                               cl_device_id *devices, int numSources,
                               const char **cfilename) {
  cl_int ciErrNum;
  cl_program program = 0;
  int i;

  char *buffer = 0;
  long length = 0;

  for (int x = 0; i < numSources; i++) {
    FILE *f = fopen(cfilename[i], "rb");

    if (f) {
      fseek(f, 0, SEEK_END);
      int size_length = ftell(f);
      fseek(f, 0, SEEK_SET);
      buffer = realloc(buffer, length + size_length);
      if (buffer) {
        fread(&buffer[length], 1, size_length, f);
      }
      fclose(f);
      length += size_length;
    }
  }

  /*  TODO add exception. */
  if (devices == NULL || nDevices < 1) {
    fprintf(stderr, "Failed to create program CL shader %s - \n %s", cfilename,
            getCLStringError(ciErrNum));
    return NULL;
  }

  cl_int status, error;
  // program = clCreateProgramWithIL(context, buffer, length, &error);
  // error = clBuildProgram(program, nDevices, devices, NULL, NULL, NULL);
  //   program = clCreateProgramWithBinary(context, nDevices, devices, length,
  //                                       buffer, &status, &error);

  /*	*/
  program = clCreateProgramWithSource(context, 1, (const char **)&buffer,
                                      &length, &ciErrNum);
  if (program == NULL || ciErrNum != CL_SUCCESS) {
    fprintf(stderr, "Failed to create program CL shader %s - \n %s", cfilename,
            getCLStringError(ciErrNum));
  }

  /*	Compile and build CL program.   */
  ciErrNum =
      clBuildProgram(program, nDevices, devices, "-cl-std=CL2.0", NULL, NULL);
  if (ciErrNum != CL_SUCCESS) {
    if (ciErrNum == CL_BUILD_PROGRAM_FAILURE) {
      char build_log[4096];
      size_t build_log_size = sizeof(build_log);
      size_t build_log_ret;
      for (i = 0; i < nDevices; i++) {
        /*	Fetch build log.	*/
        ciErrNum =
            clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                  build_log_size, build_log, &build_log_ret);

        /*	Throw error,	*/
        fprintf(stderr, "failed to compile CL shader %s - %s - %s\n", cfilename,
                (const char *)build_log, getCLStringError(ciErrNum));
      }

    } else {
      /*  */
      fprintf(stderr, "failed to compile CL shader %s - %s", cfilename,
              getCLStringError(ciErrNum));
    }
  }

  // free(source);
  return program;
}

cl_kernel createKernel(cl_program program, const char *name) {

  cl_int ciErrNum;
  cl_kernel kernel;

  kernel = clCreateKernel(program, name, &ciErrNum);

  /*  Check error.    */
  if (ciErrNum != CL_SUCCESS || !kernel) {
    fprintf(stderr, "failed to create OpeNCL kernel from program - %s\n",
            getCLStringError(ciErrNum));
    return NULL;
  }
  return kernel;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id device) {

  cl_int ciErrNum;
  cl_command_queue queue = 0;
  // cl_queue_properties pro = 0;
  //
  ///*  Create command.	*/
  // queue = clCreateCommandQueueWithProperties(context, device, &pro,
  // &ciErrNum);

  cl_command_queue_properties pro = 0;
  queue = clCreateCommandQueue(context, device, pro, &ciErrNum);

  /*  Check error.    */
  if (ciErrNum != CL_SUCCESS) {
    fprintf(stderr, "failed to create command queue - %s\n",
            getCLStringError(ciErrNum));
    return NULL;
  }
  return queue;
}

unsigned int align(const unsigned int size, const unsigned int alignment) {
  return size + (alignment - (size % alignment));
}