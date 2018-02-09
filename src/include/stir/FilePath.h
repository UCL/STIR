/*
    Copyright (C) 2018, University of Hull

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

#ifndef __stir_FILEPATH_H__
#define __stir_FILEPATH_H__

/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::FilePath
  This is a class implementing basic filesytem functionality. Parts of this class were
  copied from Functions for filename manipulations.

  Maximum effort was put in order to be cross platform.

  \author Nikos Efthimiou

*/

#include "stir/common.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <boost/format.hpp>
// N.E: Not sure I want this here. Included for the max_filename_length.
#include "stir/utilities.h"

#if defined(__OS_WIN__)
	#include <windows.h>
	 #include<sys/types.h> // required for stat.h
	#include<direct.h>
#else
	#include <unistd.h>
#endif

#include <sys/stat.h>
START_NAMESPACE_STIR

/*!
 * \ingroup buildblock
 *
 * \brief The FilePath class
 *
 * \par Description
 * This class provides basic filesystem support for STIR. It is build in a similar manner to the functions in the utilities but extends with support to create folders, check if folder exists and is writable.
 *
 * The old pathname operations are supported.
 */

class FilePath
{
public:
    FilePath();

    //!
    //! \brief FilePath
    //! \param __str Name of path
    //! \param _check Perform checks to the __str
    //!
    //! If the initial path exists in the disk _check should be left true.
    //! It will check that the path is writtable
    //! If you want to initialize to an arbitary string then set _check false.
    FilePath(const std::string &__str, bool _check = true);

    ~FilePath()
    {}

    //! Returns true if my_string points to a directory
    bool is_directory() const;
    //! Returns true if my_string points to a regular file
    //! Returns true if the path is writable. 
	//! \warning N.E: As far as I understand is only a *NIX feature. Write permissions do not exist in this level at Windows. 
    bool is_regular_file() const;
	//! On Windows the attribute INVALID_FILE_ATTRIBUTES is used. Which is not quite the same. 
    bool is_writable() const;
    //! Returns true if the path is absolute
    // This funtion has been copied from the Functions for filename manipulations
    static bool is_absolute(const std::string& _filename_with_directory);
    //! Returns true if the path already exists
    static bool exist(std::string s);
    //! Returns the current / working directory
    static std::string get_current_working_directory();
    //! Create a new folder,by FilePath, and return its path
    FilePath append(const FilePath& p);
    //! \brief Create a new folder,by string, and return its path
    //! \details This functions creates appends path p to my_string.
    //! It supports multiple new folder levels.It will try to avoid
    //! errors as permissions and non existing root path.
    FilePath append(const std::string& p);
    //! Append the separator
    static void append_separator(std::string& s);
    //! If path is no absolute then prepend the p string
    // This funtion has been copied from the Functions for filename manipulations
    void prepend_directory_name(const std::string& p);
    //! Append extension if none is present
    // This funtion has been copied from the Functions for filename manipulations
    void add_extension(const std::string& e);
    //! Replace extension
    // This funtion has been copied from the Functions for filename manipulations
    void replace_extension(const std::string& e);
    //! Get path from string
    std::string get_path() const;
    //! Get only the filename
	//! An inherent functionality from utilities is that on Windows all separators will be checked.
    std::string get_filename() const;
    //! Get the extension of the filename.
    // This function will return the the bit that is after the last dot.
    std::string get_extension() const ;

    inline bool operator==(const FilePath& other);

    inline bool operator==(const std::string& other);

    inline void operator=(const FilePath& other);

    //! \warning Possibly dangerous.
	//! When the object has been initialized with a existing path and
	//! then string in the assigment is an arbitary string this can lead to 
	//! unexpented behaviour. Hopefully, we guard sufficiently by 
	//! by performing checks again before crusial functions. 
    inline void operator=(const std::string& other);

private:
    //! Checks on wether the path exits, is writable and accessible.
    void checks() const;
    //! Split the path on char c. If c not defined then default on the current separator.
    const std::vector<std::string> split(const std::string& s, const char* c ="");

    //! Merge two strings using the correct OS separator
    std::string merge(const std::string& first, const std::string& sec);

    //! Initialize the separator based on the current OS
    inline void initSeparator();

    //! Return the position of the filename, which is the string after the last separator
    // This funtion has been copied from the Functions for filename manipulations
    std::string::size_type find_pos_of_filename() const;
    //! Find the position of the extension, if exists
    // This funtion has been copied from the Functions for filename manipulations
    std::string::size_type find_pos_of_extension() const;

protected:
    //! This is the string holding the data
    std::string my_string;
    //! The separator for the current OS.
    std::string separator;
};

END_NAMESPACE_STIR

#include "stir/FilePath.inl"

#endif


