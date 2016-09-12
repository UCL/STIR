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
#ifndef __stir_IO_GEHDF5Data_H__
#define __stir_IO_GEHDF5Data_H__
/*!
  \file
  \ingroup
  \brief

*/

#include "stir/shared_ptr.h"

#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "H5Cpp.h"
#include <string>

START_NAMESPACE_STIR

class Succeeded;

class GEHDF5Data
{
public:
    //!
    //! \brief GEHDF5Data
    //! \details Default constructor
    GEHDF5Data() {}

    //GEHDF5Data(const std::string& filename);

    void open(const std::string& filename);

    virtual ~GEHDF5Data() {};

    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    virtual shared_ptr<ExamInfo>
      get_exam_info_sptr() const;

    virtual shared_ptr<Scanner> 
      get_scanner_sptr() const;
protected:

    shared_ptr<ExamInfo> exam_info_sptr;
    shared_ptr<Scanner> scanner_sptr;
    H5::H5File file;

private:



};

END_NAMESPACE_STIR

#endif
