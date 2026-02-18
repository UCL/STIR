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

target_sources(${STIR_BUILDBLOCK_LIB} PRIVATE ${${dir_LIB_SOURCES}})

