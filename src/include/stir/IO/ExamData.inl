#include "stir/IO/ExamData.h"

START_NAMESPACE_STIR

const ExamInfo*
ExamData::get_exam_info_ptr() const
{
  return exam_info_sptr.get();
}

shared_ptr<ExamInfo>
ExamData::get_exam_info_sptr() const
{
  return exam_info_sptr;
}

END_NAMESPACE_STIR
