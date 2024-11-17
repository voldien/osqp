#pragma once
#include "opencl_helper.h"

#ifdef __cplusplus /*	C++ Environment	*/
extern "C" {
#endif

extern cl_mem cl_allocate_mem(cl_context context, const cl_mem_flags flag,
                              const size_t size);
extern cl_mem cl_allocate_calloc_mem(cl_context context, const cl_mem_flags flag,
                              const size_t size);

extern cl_mem cl_allocate_mem_data(cl_context context, const cl_mem_flags flag,
                                   void *data, const size_t size);
void cl_calloc(void **devPtr, size_t size);
extern void cl_allocate_sub_mem(const cl_mem mem, const cl_mem_flags flag,
                                const size_t number,
                                const cl_buffer_region *region, cl_mem *submem);

extern void cl_free(cl_mem mem);

extern void cl_memcpy_h2d(const cl_mem mem_dst, const void *src,
                          const size_t size);
extern void cl_memcpy_d2h(void *dst, const cl_mem mem_src, const size_t size);
extern void cl_memcpy_d2h_offset(void *dst, const cl_mem mem_src, const size_t mem_src_offset, const size_t size);


extern void cl_memcpy_d2d(cl_mem mem_dst, const cl_mem mem_src,
                          const size_t size);

extern void cl_memset(cl_mem mem_dst, const unsigned int pattern, const size_t offset,
                      const size_t size);

extern void *cl_host_map(const cl_mem mem, cl_map_flags flag,
                         const cl_buffer_region *region);
extern int cl_host_ummap(const cl_mem mem_dst, void *mapPtr);

#ifdef __cplusplus /*	C++ Environment	*/
}
#endif