#
#
set(dir utilities/UPENN)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES} ${UPENN_libgeom}
            ${UPENN_liblor} ${UPENN_liblist}
            ${UPENN_libmhdr} ${JANSSON_LIBRARY}
            z ${UPENN_libimagio} ${UPENN_libimagio++}
	    )
