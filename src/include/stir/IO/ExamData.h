 
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
    void virtual
      set_exam_info(ExamInfo const&);

protected:

      shared_ptr<ExamInfo> exam_info_sptr;

private:



};

END_NAMESPACE_STIR

#include "stir/IO/ExamData.inl"
#endif
