file(
  GLOB
  SRC_FILES
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/include/lin_sys/indirect/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/include/lin_sys/indirect/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

target_sources(OSQPLIB PRIVATE ${SRC_FILES})
target_include_directories(OSQPLIB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} include lin_sys/indirect ${OpenCL_INCLUDE_DIRS})

#math(EXPR OPENCL_VALUE "${OpenCL_VERSION_MAJOR} * 100 + ${OpenCL_VERSION_MINOR} * 10" DECIMAL)
target_compile_definitions(OSQPLIB PRIVATE CL_TARGET_OPENCL_VERSION=120 CL_USE_DEPRECATED_OPENCL_1_2_APIS)

option(OSQP_OPENCL_STATIC_LINKING "Statically link all OPENCL libraries (WARNING: This will greatly increase the file size of the library)" OFF)
mark_as_advanced(OSQP_OPENCL_STATIC_LINKING)

if(OSQP_OPENCL_STATIC_LINKING)
  message(STATUS "Statically linking OPENCL libraries")

else()
  target_link_libraries(OSQPLIB ${OpenCL_LIBRARIES} )
endif()

# ###################################
# Shader Source Files
# ###################################
FILE(GLOB_RECURSE CL_SOURCE_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/kernel/*.cl
)

# ###################################
# SPIR-V Compiler
# ###################################
FOREACH(CL ${CL_SOURCE_FILES})
	GET_FILENAME_COMPONENT(FILE_NAME ${CL} ABSOLUTE)

	SET(SPIRV "${FILE_NAME}.spv")
	SET(BC "${FILE_NAME}.bc")
	ADD_CUSTOM_COMMAND(
		OUTPUT ${SPIRV}
		COMMAND ${CMAKE_COMMAND} -E make_directory "${EXECUTABLE_OUTPUT_PATH}/Shaders/"
		COMMAND clang -Xclang -finclude-default-header -fmodules -fimplicit-module-maps  -x cl -c -target spir64 -cl-std=CL2.0 -O0  -emit-llvm -o test.bc -o ${BC} ${CL}
		COMMAND llvm-spirv ${BC} -o ${SPIRV}
		DEPENDS ${CL})
	LIST(APPEND SPIRV_BINARY_FILES ${SPIRV})
ENDFOREACH(CL)

#spirv-link -o app.spv test1.spv test2.spv

ADD_CUSTOM_TARGET(
	Shaders ALL
	DEPENDS ${SPIRV_BINARY_FILES}
)
