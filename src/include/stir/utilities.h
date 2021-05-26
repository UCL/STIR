/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2010, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_UTILITIES_H__
#define __stir_UTILITIES_H__
/*!
  \file
  \ingroup buildblock
  \brief This file declares various utility functions.

  \author Kris Thielemans
  \author PARAPET project
*/
#include "stir/common.h"
#include <stdio.h>
#include <iostream>
#include <string>

START_NAMESPACE_STIR

/******************************************************/
/*! \ingroup buildblock
  \name Functions for user input
*/
//@{
//! A function to ask a number from the user
template <class NUMBER>
inline NUMBER ask_num(const std::string& prompt, NUMBER minimum_value, NUMBER maximum_value, NUMBER default_value);

//! A function to ask a string from the user
std::string ask_string(const std::string& prompt, const std::string& default_value = "");

/*!
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
bool ask(const std::string& prompt, bool default_value);

/***** filename functions ****/

/*!
 \brief
 Asks for a filename (appending an extension if none is provided)
 and stores the string where file_in_directory_name points to.

 \deprecated Use ask_filename_with_extension(const std::string&, const std::string&);

 Example:
 \code
     char filename[max_filename_length];
     ask_filename_with_extension(filename, "Input file name ?", ".img");
 \endcode
 \warning \a file_in_directory_name has to be preallocated
       (with size \c max_filename_length)
 \warning starting or ending spaces are NOT stripped.
 */
char* ask_filename_with_extension(char* file_in_directory_name, const std::string& prompt, const std::string& default_extension);

/*!
 \brief
 Asks for a filename (appending an extension if none is provided).

 Example:
 \code
     std::string filename =
     ask_filename_with_extension("Input file name ?", ".img");
 \endcode
 \warning starting or ending spaces are NOT stripped.
 */
std::string ask_filename_with_extension(const std::string& prompt, const std::string& default_extension);

/*!
  \brief
 Asks for a filename (with default extension) and opens the stream \a s
 with \a mode giving the specifics.

 Example: open a binary input file, aborting if it is not found
 \code
    ifstream in;
    ask_filename_and_open(s, "Input file ?", ".hv", std::ios::in | std::ios::binary);
 \endcode
 Note: this function is templated to allow \a s to be of different
       types \c ifstream, \c ofstream, \c fstream
*/
// Implementation note: gcc 2.8.1 seems to have problems with default
// values when templates are around, so I overload the function
template <class FSTREAM>
void ask_filename_and_open(FSTREAM& s, const std::string& prompt, const std::string& default_extension, std::ios::openmode mode,
                           bool abort_if_failed);

//! as above, but with default \c abort_if_failed = true
template <class FSTREAM>
void inline ask_filename_and_open(FSTREAM& s, const std::string& prompt, const std::string& default_extension,
                                  std::ios::openmode mode) {
  ask_filename_and_open(s, prompt, default_extension, mode, true);
}

//@}

/******************************************************/
/*! \ingroup buildblock
  \name Functions for stream/file manipulations
*/
//@{
/*!
  \brief reads data into memory, returning a pointer to the memory

 If the file_size parameter is zero, the stream is read till EOF
 and 'file_size' is set to the number of bytes in the file.
 Otherwise 'file_size' bytes are read.

 The data is read from the current position in the stream.
 At the end of this function, the 'input' stream will be positioned
 at original_position + file_size.
 */
void* read_stream_in_memory(std::istream& input, std::streamsize& file_size);

/*!
 \brief Find number of remaining characters in the stream

 At the end of this function, the 'input' stream will be positioned
 at the original_position.
 \warning Works only properly for binary streams.
 */
std::streamsize find_remaining_size(std::istream& input);

//! opens a stream for reading binary data. Calls error() when it does not succeed.
/*! \warning probably does not work if you are not in the C-locale */
template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, const std::string& name);
//! opens a FILE for reading binary data. Calls error() when it does not succeed.
/*! \warning probably does not work if you are not in the C-locale*/
FILE*& open_read_binary(FILE*& fptr, const std::string& name);

//! opens a stream for writing binary data. Calls error() when it does not succeed.
/*!
   Templated such that it works on std::ofstream and std::fstream.
   \warning probably does not work if you are not in the C-locale
 */
template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, const std::string& name);
//! opens a FILE for writing binary data. Calls error() when it does not succeed.
/*! \warning probably does not work if you are not in the C-locale*/
FILE*& open_write_binary(FILE*& fptr, const std::string& name);

//! closes a stream without error checking.
/*!
   Templated such that it works on std::ofstream, std::ifstream and std::fstream.

   This function is only provided in case you need to write code that works with
   both std::fstream and stdio FILE.
 */
template <class FSTREAM>
inline void close_file(FSTREAM& s);

//! closes a FILE without error checking.
/*!
   This function is only provided in case you need to write code that works with
   both std::fstream and stdio FILE.
 */
void close_file(FILE*& fptr);

//@}

/*! \brief
  some large value to say how long filenames can be in
  the (deprecated) function
  ask_filename_with_extension(char *,const std::string&, const std::string&)
  \ingroup buildblock
 */
const int max_filename_length = 1000;

/******************************************************/
/*! \ingroup buildblock
  \name Functions for filename manipulations

  These functions work on different platforms, i.e. Unix, VAX, Windows. Also on
  older MacOS versions.

  \warning Functions that work on char* might be removed at some point.
*/
//@{

//! return a pointer to the start of the filename (i.e. after directory specifications)
/*! The returned pointer is between filename_with_directory and
    (filename_with_directory+strlen(filename_with_directory)+1). This
    highest value is used when it looks like a directory name.

    \deprecated use find_pos_of_filename()
    \warning This function works only with string manipulations. There is no check
    if the 'filename' part actually corresponds to a directory on disk.
*/
extern const char* find_filename(const char* const filename_with_directory);

//! return the position of the start of the filename (i.e. after directory specifications)
/*! The returned number is between 0 and filename_with_directory.size()+1. This
    highest value is used when it's a directory name.
    \warning This function works only with string manipulations. There is no check
    if the 'filename' part actually corresponds to a directory on disk.
*/
std::string::size_type find_pos_of_filename(const std::string& filename_with_directory);

//! return a std::string containing only the filename (i.e. after directory specifications)
/*!
    \warning This function works only with string manipulations. There is no check
    if the 'filename' part actually corresponds to a directory on disk.
*/
std::string get_filename(const std::string& filename_with_directory);

/*!
 \brief
 Copies the directory part from 'filename_with_directory'
 into 'directory_name' and returns the 'directory_name' pointer.
 \warning This function works only with string manipulations. There is no check
 if the 'filename' part actually corresponds to a directory on disk.

 \warning assumes that directory_name points to enough allocated space
 \deprecated Use get_directory_name(const std::string&)
*/
char* get_directory_name(char* directory_name, const char* const filename_with_directory);

//! Returns a string with the directory part from 'filename_with_directory'.
/*!
    If there was no directory information present, the return value is the string "." (i.e. the current directory).
    (It was the empty string prior to STIR 3.0).

    \warning This function works only with string manipulations. There is no check
    if the 'filename' part actually corresponds to a directory on disk.
*/
std::string get_directory_name(const std::string& filename_with_directory);

/*!
 \brief
 Checks if the filename points to an absolute location, or is
 a relative (e.g. to current directory) pathname.
 */
extern bool is_absolute_pathname(const std::string& filename_with_directory);

/*!
 \brief
 Checks if the filename points to an absolute location, or is
 a relative (e.g. to current directory) pathname.
 */
extern bool is_absolute_pathname(const char* const filename_with_directory);

/*!
 \brief
 Prepend directory_name to the filename, but only
 if <tt>!is_absolute_pathname(filename_with_directory)</tt>

 If necessary, a directory separator is inserted.
 If 'directory_name' == 0, nothing happens.
 \return a pointer to the start of the new filename
 \warning this function assumes that filename_with_directory
 points to sufficient allocated space to contain the new string.
 */
extern char* prepend_directory_name(char* filename_with_directory, const char* const directory_name);

//! find the position of the '.' of the extension
/*!
  If no '.' is found in the filename part (i.e. ignoring the
    directory name), the function returns \c std::string::npos
*/
std::string::size_type find_pos_of_extension(const std::string& file_in_directory_name);
#if 0
// terribly dangerous for memory overrun.

/*! 
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

 \deprecated
 */
extern char *
add_extension(char * file_in_directory_name, 
              const char * const extension);
#endif

//! Append extension if none present
std::string& add_extension(std::string& file_in_directory_name, const std::string& extension);

#if 0
// disabled because possible memory overrun

/*! 
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
  \deprecated
  */
extern char *
replace_extension(char *file_in_directory_name, 
                  const char * const extension);
#endif

//! Replace extension (or append if none present)
std::string& replace_extension(std::string& file_in_directory_name, const std::string& extension);

//@}

/**********************************************************************
 C-string manipulation function
***********************************************************************/
#ifndef _MSC_VER

//! make C-string uppercase
/*! \ingroup buildblock
 */
inline char* strupr(char* const str);
#else
#  define strupr _strupr
#endif

END_NAMESPACE_STIR

#include "stir/utilities.inl"

#endif // __UTILITIES_H__
