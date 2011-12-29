
#
# $Id$
#

set(dir scatter_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir_LIB_SOURCES}
	sample_scatter_points 
	single_scatter_estimate 
	single_scatter_integrals 
	scatter_detection_modelling 
        cached_single_scatter_integrals 
	scatter_estimate_for_one_scatter_point 
	upsample_and_fit_scatter_estimate 
	ScatterEstimationByBin 
)
#$(dir)_REGISTRY_SOURCES:= scatter_buildblock_registries


include(../stir_lib_target.cmake)
