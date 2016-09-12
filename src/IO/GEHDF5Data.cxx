#include "stir/IO/GEHDF5Data.h"

START_NAMESPACE_STIR

shared_ptr<Scanner>
GEHDF5Data::get_scanner_sptr() const
{
  return this->scanner_sptr;
}

shared_ptr<ExamInfo>
GEHDF5Data::get_exam_info_sptr() const
{
  return this->exam_info_sptr;
}

void
GEHDF5Data::open(const std::string& filename)
{
  this->file.openFile( filename, H5F_ACC_RDONLY );

 warning("CListModeDataGESigna: "
	  "Assuming this is GESigna, but couldn't find scan start time etc");
  this->scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));
  this->exam_info_sptr.reset(new ExamInfo);
}

END_NAMESPACE_STIR
