 
#include "stir/IO/ExamData.h"


#include <iostream>
START_NAMESPACE_STIR

ExamData::
ExamData()
{}


ExamData::ExamData(const shared_ptr<ExamInfo> &_this_exam) :
    exam_info_sptr(_this_exam)
{}


ExamData::~ExamData()
{

}

void
ExamData::set_exam_info(ExamInfo const& new_exam_info)
{
  this->exam_info_sptr.reset(new ExamInfo(new_exam_info));
}

END_NAMESPACE_STIR
