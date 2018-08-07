/*
    Copyright (C) 2016 University College London
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
#ifndef __stir_IO_HDF5Wrapper_H__
#define __stir_IO_HDF5Wrapper_H__
/*!
  \file
  \ingroup
  \brief

*/

#include "stir/shared_ptr.h"
#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

#include "H5Cpp.h"

#include <string>

START_NAMESPACE_STIR

class HDF5Wrapper
{
public:
    //!
    //! \brief GEHDF5Data
    //! \details Default constructor
    explicit HDF5Wrapper();

    explicit HDF5Wrapper(const std::string& filename);

    Succeeded open(const std::string& filename);

    static bool check_GE_signature(const std::string filename);

    ~HDF5Wrapper() {}

    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    shared_ptr<ExamInfo>
    get_exam_info_sptr() const;

    shared_ptr<Scanner>
    get_scanner_sptr() const;

    //! \obsolete
    inline const H5::H5File& get_file() const
    { return file; }

protected:

    //    shared_ptr<ExamInfo> exam_info_sptr;

    Succeeded initialise_scanner_from_HDF5();

    shared_ptr<Scanner> scanner_sptr;
private:

    H5::H5File file;

    bool is_signa = false;

};

END_NAMESPACE_STIR

#endif
