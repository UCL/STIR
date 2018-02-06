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

  \author Nikos Efthimiou

*/

#include "stir/common.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <boost/format.hpp>

#include <sys/stat.h>
#include <unistd.h>
#if defined(__OS_WIN__)
#include <windows.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif
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
    bool is_regfile() const;
    //! Returns true if the path is writable
    bool is_writable() const;
    //! Returns true if the path is absolute
    static bool is_absolute(const std::string _filename_with_directory);
    //! Returns true if the path already exists
    static bool exist(std::string s);

    static std::string get_current_dirname();
    //! Create a new folder,by FilePath, and return its path
    FilePath append(FilePath p);
    //! Create a new folder,by string, and return its path
    FilePath append(std::string p);
    //! Append the separator
    static void append_separator(std::string& s);
    //! If path is no absolute then prepend the p string
    void prepend_directory_name(std::string p);

    //! Append extension if none is present
    void add_extension(const std::string e);
    //! Replace extension
    void replace_extension(const std::string e);
    //! Get path from string
    std::string get_path() const;

    std::string
    get_filename() const;

    std::string
    get_extension() const ;

    inline bool operator==(const FilePath& other) {
        if (this->my_string==other.my_string)
            return true;
        else
            return false;
    }

    inline bool operator==(const std::string& other) {
        if (this->my_string==other)
            return true;
        else
            return false;
    }

    inline void operator=(const FilePath& other) {
        this->my_string = other.my_string;
        this->separator = other.separator;
    }

    //! \warning Possibly dangerous... when real paths are used.
    inline void operator=(const std::string& other) {
        this->my_string = other;
        this->initSeparator();
    }

private:
    //! Checks on wether the path exits, is writable and accessible.
    void checks() const;
    //! Split the path on char c. If c not defined then default on the current separator.
    const std::vector<std::string> split(const std::string& s, const char* c ="");

    //! Merge two strings using the correct OS separator
    std::string merge(std::string first, std::string sec);

    //! Initialize the separator based on the current OS
    inline void initSeparator()
    {
#if defined(__OS_VAX__)
        separator = ":";
#elif defined(__OS_WIN__)
        separator = "\\";
#elif defined(__OS_MAC__)
        separator = ":" ;
#else // defined(__OS_UNIX__)
        separator = "/" ;
#endif
    }
    //! Return the position of the filename, which is the string after the last separator
    std::string::size_type find_pos_of_filename(std::string _s = "") const;
    //! Find the position of the extension, if exists
    std::string::size_type find_pos_of_extension() const;

protected:
    //! This is the string holding the data
    std::string my_string;
    //! The separator for the current OS.
    std::string separator;
};

END_NAMESPACE_STIR

#endif


