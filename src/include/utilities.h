//
// $Id$: $Date$
//
#ifndef __UTILITIES_H__
#define  __UTILITIES_H__


#include "pet_common.h"
#include "sinodata.h"


// ask_PSOV_details asks for filename etc. and returns a PSOV to play with.
PETSinogramOfVolume ask_PSOV_details(iostream * p_in_stream,
				     const bool on_disk = true);


// Reads data into memory, returning a pointer to the memory
// If the file_size parameter is zero, the stream is read till EOF
// and 'file_size' is set to the number of bytes in the file. 
// Otherwise 'file_size' bytes are read.
// The data is read from the current position in the stream.
// At the end of this function, the 'input' stream will be positioned 
// at original_position + file_size.
void * read_stream_in_memory(istream& input, unsigned long& file_size);

// Find number of remaining characters in the stream
// (works only properly for binary streams)
// At the end of this function, the 'input' stream will be positioned 
// at the original_position
streamsize find_remaining_size (istream& input);

#endif // __UTILITIES_H__
