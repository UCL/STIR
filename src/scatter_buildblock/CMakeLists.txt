
#
#

set(dir scatter_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir_LIB_SOURCES}
	sample_scatter_points.cxx
	single_scatter_estimate.cxx
	single_scatter_integrals.cxx
	scatter_detection_modelling.cxx
        cached_single_scatter_integrals.cxx
	scatter_estimate_for_one_scatter_point.cxx
	upsample_and_fit_scatter_estimate.cxx
	ScatterEstimation.cxx
	CreateTailMaskFromACFs.cxx
	ScatterSimulation.cxx
	SingleScatterSimulation.cxx
)
#$(dir)_REGISTRY_SOURCES:= scatter_buildblock_registries


include(stir_lib_target)
target_link_libraries(scatter_buildblock PUBLIC recon_buildblock PRIVATE fmt)
