#
#

set(dir modelling_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	KineticModel.cxx
	PatlakPlot.cxx
	ParametricDiscretisedDensity.cxx
)

#$(dir)_REGISTRY_SOURCES:= modelling_registries

include(stir_lib_target)

target_link_libraries(modelling_buildblock PUBLIC buildblock IO)
