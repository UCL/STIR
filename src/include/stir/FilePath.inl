/*
    Copyright (C) 2018, University of Hull

    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

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

inline std::string FilePath::get_string() const
{return my_string;}

END_NAMESPACE_STIR
