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

// KT 06/11/98 moved from pet_common.h

/*****************************************************
 ask*() functions for user input
*****************************************************/

// A function to ask a number from the user 
// KT 21/05/98 added, replaces asknr from gen.c, it can be used for any numeric type
template <class CHARP, class NUMBER>
NUMBER ask_num (CHARP str,NUMBER minimum_value, NUMBER maximum_value, NUMBER default_value)
{ 
  
  while(1)
  { 
    char input[30];

    cerr << "\n" << str 
         << "[" << minimum_value << "," << maximum_value 
	 << " D:" << default_value << "]: ";
    fgets(input,30,stdin);
    istrstream ss(input);
    NUMBER value = default_value;
    ss >> value;
    if ((value>=minimum_value) && (maximum_value>=value))
      return value;
    cerr << "\nOut of bounds. Try again.";
  }
}

// KT 30/07/98 added, replaces ask from gen.c
template <class CHARP>
bool ask (CHARP str, bool default_value)
{ 
  
  char input[30];
  
  cerr << "\n" << str 
       << " [Y/N D:" 
       << (default_value ? 'Y' : 'N') 
       << "]: ";
  fgets(input,30,stdin);
  if (strlen(input)==0)
    return default_value;
  char answer = input[0];
  if (default_value==true)
  {
    if (answer=='N' || answer == 'n')
      return false;
    else
      return true;
  }
  else
  {
    if (answer=='Y' || answer == 'y')
      return true;
    else
      return false;
    
  }
}

/*****************************************************
 functions for opening binary streams
*****************************************************/
// KT 22/05/98 added next 2
#include <fstream>


// KT 09/08/98 added const
// KT 09/10/98 use correct syntax for const pointers
template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, 
				  const char * const name)
{
#if 0
  //KT 30/07/98 The next lines are only necessary (in VC 5.0) when importing 
  // <fstream.h>. We use <fstream> now, so they are disabled.

  // Visual C++ does not complain when opening a nonexisting file for reading,
  // unless using ios::nocreate
  s.open(name, ios::in | ios::binary | ios::nocreate); 
#else
  s.open(name, ios::in | ios::binary); 
#endif
  if (s.fail() || s.bad())
    { PETerror("Error opening file\n"); Abort(); }
  return s;
}

// KT 09/08/98 added const
// KT 09/10/98 use correct syntax for const pointers
template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, 
				  const char * const name)
{
    s.open(name, ios::out | ios::binary); 
    if (s.fail() || s.bad())
    { PETerror("Error opening file\n"); Abort(); }
    return s;
}


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
