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
  \brief

  \author Nikos Efthimiou

*/

#include "stir/common.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <boost/format.hpp>

#include <sys/stat.h>
#include <unistd.h>

START_NAMESPACE_STIR

class FilePath
{
public:
    FilePath();

    FilePath(const std::string &__str, bool _check = true);

    ~FilePath()
    {}

    bool is_directory() const;

    bool is_regfile() const;

    bool is_writable() const;

    static bool is_absolute(const std::string _filename_with_directory);

    static bool exist(std::string s);

    FilePath append(FilePath p);

    FilePath append(std::string p);

    void prepend_directory_name(std::string p);

    //! Append extension if none is present
    void
    add_extension(const std::string e);

    void
    replace_extension(const std::string e);

    std::string
    get_path() const;

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

    //! Possibly dangerous... when real paths are used.
    inline void operator=(const std::string& other) {
           this->my_string = other;
           this->initSeparator();
        }

private:

    void checks() const;

    const std::vector<std::string> split(const std::string& s, const char* c);

    std::string merge(std::string first, std::string sec);

    inline void initSeparator()
    {
#if defined(__OS_WIN__)
        separator = "\\";
#else // defined(__OS_UNIX__)
        separator = "/" ;
#endif
    }

    std::string::size_type
    find_pos_of_filename() const;

    std::string::size_type
    find_pos_of_extension() const;

protected:
       std::string my_string;
       std::string separator;

};

END_NAMESPACE_STIR

#endif


