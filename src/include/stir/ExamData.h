/*
    Copyright (C) 2016, 2018, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
    void set_exam_info_sptr(shared_ptr<const ExamInfo>  new_exam_info_sptr);

protected:

      shared_ptr<const ExamInfo> exam_info_sptr;

private:



};

END_NAMESPACE_STIR

#endif
