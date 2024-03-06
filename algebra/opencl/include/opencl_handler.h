#pragma once

#include "osqp_api_types.h"

extern OSQPInt opencl_init(OSQPInt device);

extern void opencl_release(void);