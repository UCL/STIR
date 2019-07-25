
# For GPU projectors, we'll need both CUDA and NiftyPET
# If either are missing, return

# Try to find CUDA
find_package(CUDA QUIET)
if (NOT CUDA_FOUND)
  return()
else()
  message(STATUS "CUDA Version: ${CUDA_VERSION}")
endif()

# Try to find NiftyPET
# commented out for now as this functionality does 
# not currently exist in NiftyPET
find_package(NiftyPET QUIET)
if (NOT NiftyPET_FOUND)
  return()
endif()

