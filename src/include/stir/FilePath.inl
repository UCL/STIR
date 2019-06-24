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

/*!
  \file
  \ingroup buildblock

  \brief Implementations of inline functions for class stir::FilePath

  \author Nikos Efthimiou
*/
START_NAMESPACE_STIR

inline bool FilePath::operator==(const FilePath& other) {
    if (this->my_string==other.my_string)
        return true;
    else
        return false;
}

inline bool FilePath::operator==(const std::string& other) {
    if (this->my_string==other)
        return true;
    else
        return false;
}

inline void FilePath::operator=(const FilePath& other) {
    this->my_string = other.my_string;
    this->separator = other.separator;
}

//! \warning Possibly dangerous... when real paths are used.
inline void FilePath::operator=(const std::string& other) {
    this->my_string = other;
    this->initSeparator();
}


inline void FilePath::initSeparator()
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

END_NAMESPACE_STIR
