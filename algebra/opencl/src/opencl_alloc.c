#include "opencl_alloc.h"
#include "osqp_api_types.h"
#include <CL/cl.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>

extern OpenCL_Handle_t *handle;

cl_mem cl_allocate_mem(cl_context context, const cl_mem_flags flag,
                       const size_t size) {
  cl_int err;
  cl_mem mem = clCreateBuffer(context, flag, size, NULL, &err);
  assert(err == CL_SUCCESS);
  return mem;
}

cl_mem cl_allocate_calloc_mem(cl_context context, const cl_mem_flags flag,
                              const size_t size) {
  cl_int err;
  cl_mem mem = clCreateBuffer(context, flag, size, NULL, &err);
  assert(err == CL_SUCCESS);
  cl_memset(mem, 0, 0, size);
  return mem;
}

cl_mem cl_allocate_mem_data(cl_context context, const cl_mem_flags flag,
                            void *data, const size_t size) {
  cl_int err;
  cl_mem mem = clCreateBuffer(context, flag, size, data, &err);
  assert(err == CL_SUCCESS);
  return mem;
}

void cl_allocate_sub_mem(const cl_mem mem, const cl_mem_flags flag,
                         const size_t number, const cl_buffer_region *region,
                         cl_mem *submem) {
  cl_int err;
  for (size_t i = 0; i < number; i++) {
    submem[i] = clCreateSubBuffer(mem, flag, CL_BUFFER_CREATE_TYPE_REGION,
                                  &region[i], &err);
    assert(err == CL_SUCCESS);
  }
}

void cl_free(cl_mem mem) {
  cl_int err = clReleaseMemObject(mem);
  assert(err == CL_SUCCESS);
}

void cl_memcpy_h2d(const cl_mem mem_dst, const void *src, const size_t size) {
  if (mem_dst && size > 0 && src) {
    cl_int err;
    err = clEnqueueWriteBuffer(handle->queue, mem_dst, CL_TRUE, 0, size, src, 0,
                               NULL, NULL);
    assert(err == CL_SUCCESS);
  }
}

void cl_memcpy_d2h(void *dst, const cl_mem mem_src, const size_t size) {
  cl_int err;
  err = clEnqueueReadBuffer(handle->queue, mem_src, CL_TRUE, 0, size, dst, 0,
                            NULL, NULL);
  assert(err == CL_SUCCESS);
}

void cl_memcpy_d2h_offset(void *dst, const cl_mem mem_src,
                          const size_t mem_src_offset, const size_t size) {
  cl_int err;
  err = clEnqueueReadBuffer(handle->queue, mem_src, CL_TRUE, mem_src_offset,
                            size, dst, 0, NULL, NULL);
  assert(err == CL_SUCCESS);
}

void cl_memcpy_d2d(cl_mem mem_dst, const cl_mem mem_src, const size_t size) {
  if (mem_dst && size > 0 && mem_src) {
    cl_int err;
    err = clEnqueueCopyBuffer(handle->queue, mem_src, mem_dst, 0, 0, size, 0,
                              NULL, NULL);
    assert(err == CL_SUCCESS);
  }
}

void cl_memset(cl_mem mem_dst, const unsigned int pattern, const size_t offset,
               const size_t size) {
  cl_int err =
      clEnqueueFillBuffer(handle->queue, mem_dst, &pattern, sizeof(pattern),
                          offset, size, 0, NULL, NULL);
}

void *cl_host_map(const cl_mem mem, cl_map_flags flag,
                  const cl_buffer_region *region) {
  cl_int err;
  void *ptr =
      clEnqueueMapBuffer(handle->queue, mem, CL_FALSE, flag, region->origin,
                         region->size, 0, NULL, NULL, &err);
  assert(err == CL_SUCCESS);
  assert(ptr != NULL);
  return ptr;
}

int cl_host_ummap(const cl_mem mem_dst, void *mapPtr) {

  cl_int err =
      clEnqueueUnmapMemObject(handle->queue, mem_dst, mapPtr, 0, NULL, NULL);
  assert(err == CL_SUCCESS);
  return err;
}
