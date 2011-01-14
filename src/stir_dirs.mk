#
# $Id$
#
# This file contains a list of all subdirectories in STIR
# that will be compiled when using the Makefile

LIBDIRS += buildblock recon_buildblock display IO \
	data_buildblock \
	numerics_buildblock \
	eval_buildblock Shape_buildblock \
	listmode_buildblock \
	modelling_buildblock \
	scatter_buildblock \
	iterative/OSMAPOSL \
	iterative/OSSPS \
	analytic/FBP2D \
	analytic/FBP3DRP

EXEDIRS += utilities recon_test \
	listmode_utilities \
	modelling_utilities \
	scatter_utilities \
	utilities/ecat \
	iterative/OSMAPOSL \
	iterative/POSMAPOSL \
	iterative/OSSPS \
	iterative/POSSPS \
	analytic/FBP2D \
	analytic/FBP3DRP \
	SimSET \
	scripts

TESTDIRS += test recon_test test/numerics test/modelling
