//
//
/*!
  \file
  \ingroup utilities

  \brief This program takes a projection data from one type and converts it to another type.
  \author Parisa Khateri

*/
/*
*/
#include "stir/ProjData.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/Bin.h"

#include <fstream>
#include <iostream>
#include "stir/ProjDataFromStream.h"
#include "stir/Viewgram.h"
#include "stir/IO/read_from_file.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include "stir/LORCoordinates.h"

#include "stir/GeometryBlocksOnCylindrical.h"
#include "stir/DetectionPosition.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/DetectorCoordinateMap.h"
#include <boost/make_shared.hpp>
#include "stir/CPUTimer.h"
#include "stir/shared_ptr.h"


#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR



int main(int argc, char *argv[])
{
  CPUTimer timer0;

  if(argc<4)
  {
    cerr<<"Usage: " << argv[0] << " output_filename input_filename template_blk_projdata\n";
    exit(EXIT_FAILURE);
  }
  std::string output_filename=argv[1];
  shared_ptr<ProjData> in_pd_ptr = ProjData::read_from_file(argv[2]);
  shared_ptr<ProjData> template_pd_ptr = ProjData::read_from_file(argv[3]);

  shared_ptr<ProjDataInfo> in_pdi_ptr(in_pd_ptr->get_proj_data_info_sptr()->clone());
  shared_ptr<ProjDataInfo> out_pdi_ptr(template_pd_ptr->get_proj_data_info_sptr()->clone());
  ProjDataInterfile out_proj_data(template_pd_ptr->get_exam_info_sptr(), out_pdi_ptr, output_filename+".hs");
  write_basic_interfile_PDFS_header(output_filename+".hs", out_proj_data);

  timer0.start();

  assert(in_pdi_ptr->get_min_segment_num()==-1*in_pdi_ptr->get_max_segment_num());
  for (int seg=in_pdi_ptr->get_min_segment_num(); seg<=in_pdi_ptr->get_max_segment_num();++seg)
  {
    std::cout<<"seg_num = "<<seg<<"\n";
    // keep sinograms out of the loop to avoid reallocations
    // initialise to something because there's no default constructor
    Viewgram<float> viewgram_blk = out_proj_data.get_empty_viewgram(out_proj_data.get_min_view_num(),seg);
    Viewgram<float> viewgram_cyl = in_pd_ptr->get_empty_viewgram(in_pd_ptr->get_min_view_num(),seg);

    for(int view=in_pdi_ptr->get_min_view_num(); view<=in_pdi_ptr->get_max_view_num();++view)
    {
      viewgram_blk = out_proj_data.get_empty_viewgram(view,seg);
      viewgram_cyl = in_pd_ptr->get_viewgram(view,seg);

      for(int ax=in_pdi_ptr->get_min_axial_pos_num(seg); ax<=in_pdi_ptr->get_max_axial_pos_num(seg);++ax)
      {
        for(int tang=in_pdi_ptr->get_min_tangential_pos_num(); tang<=in_pdi_ptr->get_max_tangential_pos_num(); ++tang)
        {
          viewgram_blk[ax][tang] = viewgram_cyl[ax][tang];
        }
      }
      if (!(out_proj_data.set_viewgram(viewgram_blk)== Succeeded::yes))
        warning("Error set_segment for projdata_symm %d\n", seg);
    }
  }

  timer0.stop();
  std::cerr << "\nConverting from cylindrical projdata to block took " << timer0.value() << "s CPU time.\n\n";

  return EXIT_SUCCESS;
}
