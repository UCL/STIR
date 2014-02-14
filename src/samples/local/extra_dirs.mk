#
#

LIBDIRS += local/buildblock local/recon_buildblock 

EXEDIRS += local/utilities \

-include local/extra_dirs_$(USER).mk

TESTDIRS += local/test 
