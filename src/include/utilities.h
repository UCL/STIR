//
// $Id$: $Date$
//
#ifndef __UTILITIES_H__
#define  __UTILITIES_H__


#include "pet_common.h"
#include "sinodata.h"
#include "imagedata.h"

// KT 09/10/98 new
// ask_image_details asks for filename etc, and returns an image
PETImageOfVolume ask_image_details();


// ask_PSOV_details asks for filename etc. and returns a PSOV to play with.
PETSinogramOfVolume ask_PSOV_details(iostream * p_in_stream,
				     const bool on_disk = true);


// read_stream_in_memory
// reads data into memory, returning a pointer to the memory
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

// KT 21/10/98 new

/**************************************************************************
 Some functions to manipulate (and ask for) filenames.
***************************************************************************/

// some large value to says how long filenames can be in the 
// functions below
const int max_filename_length = 1000;

// return a pointer to the start of the filename 
// (i.e. after directory specifications)
extern const char * const 
find_filename(const char * const filename_with_directory);

// Append 'extension' to 'filename_with_directory'
// if no '.' is found in 'filename_with_directory'
// (excluding the directory part)
// Returns the 'filename_with_directory' pointer.
// Example (on Unix):
//     char filename[max_filename_length] = "dir.name/filename";
//     add_extension(filename, ".img");
//   results in 'filename' pointing to "dir.name/filename.img"
extern char *
add_extension(char * file_in_directory_name, 
	      const char * const extension);


// Asks for a filename (appending an extension if none is provided)
// and stores where file_in_directory_name points to.
// Example:
//     char filename[max_filename_length];
//     ask_filename_with_extension(filename, "Input file name ?", ".img");
// Note: 'file_in_directory_name' has to be preallocated 
//       (with size max_filename_length)
// Restriction: the filename cannot contain spaces
extern char *
ask_filename_with_extension(char *file_in_directory_name, 
  		            const char * const prompt,
			    const char * const default_extension);

// Asks for a filename (with default extension) and opens the stream 's'
// with 'mode' giving the specifics. 
// Example: open a binary input file, aborting if it is not found
//    ifstream in;
//    ask_filename_and_open(s, "Input file ?", ".hv", ios::in | ios::binary);
// Note: this function is templated to allow 's to be of different
//       types ifstream, ofstream, fstream
// Implementation note: gcc 2.8.1 seems to have problems with default
// values when templates are around, so I overload the function

template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);

template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode)
{ 
  ask_filename_and_open(s, prompt, default_extension, mode, true);
}



#endif // __UTILITIES_H__
