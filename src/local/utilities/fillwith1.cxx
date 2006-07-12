//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief A utility that just fills the projection data with 1

  \author Kris Thielemans

  $Date$
  $Revision$
*/


#include "stir/ProjData.h"
#include "stir/SegmentByView.h"
#include "stir/Succeeded.h"

#include <iostream> 



USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=2) 
  {
    std::cerr<<"Usage: " << argv[0] << " projdata_file\n"
       	<< std::endl; 
  }

  shared_ptr<ProjData> projdata_ptr = 
  ProjData::read_from_file(argv[1], ios::in|ios::out);
  

  for (int segment_num=projdata_ptr->get_min_segment_num();
       segment_num<=projdata_ptr->get_max_segment_num();
       ++segment_num)
    {
      SegmentByView<float> segment = projdata_ptr->get_empty_segment_by_view(segment_num,false);
      segment.fill(1);
      if (projdata_ptr->set_segment(segment) != Succeeded::yes)
	error("fillwith1 failed writing segment %d of file %s",
	      segment_num, argv[1]);
    }
  return EXIT_SUCCESS;

}
