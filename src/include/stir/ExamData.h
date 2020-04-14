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
  \ingroup buildblock
  \brief declaration of stir::ExamData

  \author Nikos Efthimiou
*/

#include "stir/shared_ptr.h"
#include <vector>
#include "stir/ExamInfo.h"

START_NAMESPACE_STIR

class Succeeded;

/*! 
  \brief base class for data objects such as ProjData etc
  \ingroup buildblock

  Provides an ExamInfo member.
*/
class ExamData
{
public:
    //!
    //! \brief ExamData
    //! \details Default constructor
    ExamData();

    ExamData(const shared_ptr < const ExamInfo > & _this_exam);

    virtual ~ExamData();

    virtual const ExamInfo&
      get_exam_info() const;
    //! Get shared pointer to exam info
    virtual shared_ptr<const ExamInfo>
      get_exam_info_sptr() const;
    //! change exam info
    /*! This will allocate a new ExamInfo object and copy the data in there. */
    virtual void
      set_exam_info(ExamInfo const&);

protected:

      shared_ptr<const ExamInfo> exam_info_sptr;

private:



};

END_NAMESPACE_STIR

#endif
