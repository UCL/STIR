/*
    Copyright (C) 2016, 2018, University College London
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
#ifndef __stir_IO_ExamData_H__
#define __stir_IO_ExamData_H__
/*!
  \file
  \ingroup
  \brief

  \author Nikos Efthimiou
*/

#include "stir/shared_ptr.h"
#include <vector>
#include "stir/ExamInfo.h"

START_NAMESPACE_STIR

class Succeeded;

class ExamData
{
public:
    //!
    //! \brief ExamData
    //! \details Default constructor
    ExamData();

    ExamData(const shared_ptr < ExamInfo > & _this_exam);

    virtual ~ExamData();

    inline virtual const ExamInfo&
      get_exam_info() const;
    //! Get pointer to exam info
    inline virtual const ExamInfo*
      get_exam_info_ptr() const;
    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    inline virtual shared_ptr<ExamInfo>
      get_exam_info_sptr() const;
    //! change exam info
    /*! This will allocate a new ExamInfo object and copy the data in there. */
	virtual void
      set_exam_info(ExamInfo const&);

protected:

      shared_ptr<ExamInfo> exam_info_sptr;

private:



};

END_NAMESPACE_STIR

#include "stir/IO/ExamData.inl"
#endif
