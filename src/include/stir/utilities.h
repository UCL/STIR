//
// $Id$
//
#ifndef __stir_UTILITIES_H__
#define  __stir_UTILITIES_H__
/*!
  \file 
  \ingroup buildblock
  \brief This file declares various utility functions.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/common.h"

#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::iostream;
using std::istream;
using std::streamsize;
using std::string;
#endif

START_NAMESPACE_STIR


/*! \ingroup buildblock
  \brief reads data into memory, returning a pointer to the memory

 If the file_size parameter is zero, the stream is read till EOF
 and 'file_size' is set to the number of bytes in the file. 
 Otherwise 'file_size' bytes are read.

 The data is read from the current position in the stream.
 At the end of this function, the 'input' stream will be positioned 
 at original_position + file_size.
 */
void * read_stream_in_memory(istream& input, unsigned long& file_size);

/*! \ingroup buildblock
 \brief Find number of remaining characters in the stream

 At the end of this function, the 'input' stream will be positioned 
 at the original_position.
 \warning Works only properly for binary streams.
 */
streamsize find_remaining_size (istream& input);

/*****************************************************
 ask*() functions for user input
*****************************************************/

//! A function to ask a number from the user 
/*!  \ingroup buildblock  
*/
template <class NUMBER>
inline NUMBER 
ask_num (const string& prompt,
	 NUMBER minimum_value, 
	 NUMBER maximum_value, 
	 NUMBER default_value);

//! A function to ask a string from the user 
/*!  \ingroup buildblock
 */
string
ask_string(const string& prompt, const string& default_value = "");

/*!  \ingroup buildblock
  \brief A function to ask a yes/no question from the user
  
  \param prompt a text string which is supposed to be a question
  \param default_value When set to \c true , the default is Yes.

  The question is currently presented as
  \verbatim
  prompt [Y/N D:default_value]: 
  \endverbatim
  Simply pressing 'enter' will select the default value. Otherwise, 
  the first charachter of the response will be checked against 
  y/Y/n/N to determine the return value. If it is none of these,
  the question will be asked again.
  */
bool 
ask (const string& prompt, bool default_value);

/*****************************************************
 functions for opening binary streams
*****************************************************/

//! opens a stream for reading. Calls error() when it does not succeed.
/*!  \ingroup buildblock
 */
template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, 
				  const string& name);
//! opens a stream for writing. Calls error() when it does not succeed.
/*!  \ingroup buildblock
 */
template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, 
				  const string& name);



/**************************************************************************
 Some functions to manipulate (and ask for) filenames.
***************************************************************************/

/*! \brief
  some large value to say how long filenames can be in  ask_filename_with_extension() and ask_filename_and_open()
  \ingroup buildblock
 */
const int max_filename_length = 1000;

//! return a pointer to the start of the filename (i.e. after directory specifications)
/*! \ingroup buildblock
    The returned pointer is between filename_with_directory and 
    (filename_with_directory+strlen(filename_with_directory)+1). This
    highest value is used when it's a directory name.
*/
extern const char * const 
find_filename(const char * const filename_with_directory);

//! return the position of the start of the filename (i.e. after directory specifications)
/*! \ingroup buildblock
    The returned number is between 0 and filename_with_directory.size()+1. This
    highest value is used when it's a directory name.
*/
string::size_type
find_pos_of_filename(const string& filename_with_directory);


/*! \ingroup buildblock
 \brief
 Copies the directory part from 'filename_with_directory'
 into 'directory_name' and returns the 'directory_name' pointer.

\warning assumes that directory_name points to enough allocated space
*/
char *
get_directory_name(char *directory_name, 
		   const char * const filename_with_directory);

//! Returns a string with the directory part from 'filename_with_directory'.
/*! \ingroup buildblock
 */
string
get_directory_name(const string& filename_with_directory);

/*! \ingroup buildblock
 \brief
 Checks if the filename points to an absolute location, or is
 a relative (e.g. to current directory) pathname.
 */
extern bool
is_absolute_pathname(const string& filename_with_directory);

/*! \ingroup buildblock
 \brief
 Checks if the filename points to an absolute location, or is
 a relative (e.g. to current directory) pathname.
 */
extern bool
is_absolute_pathname(const char * const filename_with_directory);


/*! \ingroup buildblock
 \brief
 Prepend directory_name to the filename, but only
 if <tt>!is_absolute_pathname(filename_with_directory)</tt>

 If necessary, a directory separator is inserted.
 If 'directory_name' == 0, nothing happens.
 \return a pointer to the start of the new filename
 \warning this function assumes that filename_with_directory 
 points to sufficient allocated space to contain the new string.
 */
extern char *
prepend_directory_name(char * filename_with_directory, 
		       const char * const directory_name);

//! find the position of the '.' of the extension
/*! \ingroup buildblock
  If no '.' is found in the filename part (i.e. ignoring the
    directory name), the function returns \c string::npos
*/
string::size_type
find_pos_of_extension(const string& file_in_directory_name);

/*! \ingroup buildblock
\brief
 Append \a extension to \a filename_with_directory
 if no '.' is found in \a filename_with_directory
 (excluding the directory part)
 \return the \a filename_with_directory pointer.

 Example (on Unix):
 \code
     char filename[max_filename_length] = "dir.name/filename";
     add_extension(filename, ".img");
 \endcode
   results in 'filename' pointing to "dir.name/filename.img"

 On Windows systems, both forward and backslash can be used.
 */
extern char *
add_extension(char * file_in_directory_name, 
	      const char * const extension);

//! Append extension when input parameters are strings
/*! \ingroup buildblock
 */
string& 
add_extension(string& file_in_directory_name, 
	      const string& extension);

/*! \ingroup buildblock
  \brief
  Replace extension in \a filename_with_directory with \a extension.

  if no extension is found in 'filename_with_directory',
 'extension' is appended.
 \return  the 'filename_with_directory' pointer.

 Example (on Unix):
 \code
     char filename[max_filename_length] = "dir.name/filename.v";
     replace_extension(filename, ".img");
 \endcode
  results in 'filename' pointing to "dir.name/filename.img"
  */
extern char *
replace_extension(char *file_in_directory_name, 
 	          const char * const extension);

//! Replace extension when input parameters are strings
/*! \ingroup buildblock
 */
string& 
replace_extension(string& file_in_directory_name, 
		  const string& extension);
		   
/*! \ingroup buildblock
 \brief
 Asks for a filename (appending an extension if none is provided).

 Example:
 \code
     string filename =
     ask_filename_with_extension("Input file name ?", ".img");
 \endcode
 \warning starting or ending spaces are NOT stripped.
 */
string
ask_filename_with_extension(const string& prompt,
			    const string& default_extension);


/*! \ingroup buildblock
 \brief
 Asks for a filename (appending an extension if none is provided)
 and stores the string where file_in_directory_name points to.

 \deprecated
 Example:
 \code
     char filename[max_filename_length];
     ask_filename_with_extension(filename, "Input file name ?", ".img");
 \endcode
 \warning \a file_in_directory_name has to be preallocated 
       (with size \c max_filename_length)
 \warning starting or ending spaces are NOT stripped.
 */
extern char *
ask_filename_with_extension(char *file_in_directory_name, 
  		            const string& prompt,
			    const string& default_extension);

/*! \ingroup buildblock
  \brief
 Asks for a filename (with default extension) and opens the stream 's'
 with 'mode' giving the specifics. 

 Example: open a binary input file, aborting if it is not found
 \code
    ifstream in;
    ask_filename_and_open(s, "Input file ?", ".hv", ios::in | ios::binary);
 \endcode
 Note: this function is templated to allow 's to be of different
       types ifstream, ofstream, fstream
*/
// Implementation note: gcc 2.8.1 seems to have problems with default
// values when templates are around, so I overload the function
template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);

//! as above, but with default \c abort_if_failed = true
/*! \ingroup buildblock
 */
template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode)
{ 
  ask_filename_and_open(s, prompt, default_extension, mode, true);
}

/**********************************************************************
 C-string manipulation function
***********************************************************************/
#ifndef _MSC_VER

//! make C-string uppercase
/*! \ingroup buildblock
*/
inline char *strupr(char * const str);
#else
#define strupr _strupr
#endif

END_NAMESPACE_STIR

#include "stir/utilities.inl"

#endif // __UTILITIES_H__
