//
//
/*
    
*/
/*!

  \file
  \ingroup utilities
  \brief Main program for trim_projdata

  \author Parisa Khateri


   \par Usage:
   \code
   trim_projdata [-t num_tang_poss_to_trim] \
        output_filename input_projdata_name 
   \endcode
   \param num_tang_poss_to_trim has to be smaller than the available number
      of tangential positions.

  \par Example:
  \code
  trim_projdata -t 10 out in.hs
  \endcode
*/
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include <string>
#include <string.h>
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/ProjDataInfoBlocksOnCylindrical.h"
#include "stir/ProjDataInfoGeneric.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include <fstream>
#include <algorithm>


#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  int num_tang_poss_to_trim = 0;
  if (argc>1 && strcmp(argv[1], "-t")==0)
    {
      num_tang_poss_to_trim =  atoi(argv[2]);
      argc -= 2; argv += 2;
    }
  if (argc > 5 || argc < 3 )
    {
      cerr << "Usage:\n"
	   << argv[0] << " [-t num_tang_poss_to_trim] \\\n"
	   << "\toutput_filename input_projdata_name \\\n"
	   << "num_tang_poss_to_trim has to be smaller than the available number\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);
  
  
  if (in_projdata_ptr->get_num_tangential_poss() <=
      num_tang_poss_to_trim)
    error("trim_projdata: too large number of tangential positions to trim (%d)\n",
    num_tang_poss_to_trim);
  
  if (in_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry() ==
      "Cylindrical")
  {
    const ProjDataInfoCylindrical * const in_projdata_info_cyl_ptr =
      dynamic_cast<ProjDataInfoCylindrical const * >(in_projdata_ptr->get_proj_data_info_sptr());
    if (in_projdata_info_cyl_ptr== NULL)
      {
        error("error converting to cylindrical projection data\n");
      }
    ProjDataInfoCylindrical * out_projdata_info_cyl_ptr = 
      dynamic_cast<ProjDataInfoCylindrical * >
      (in_projdata_info_cyl_ptr->clone());

    out_projdata_info_cyl_ptr->
      set_num_tangential_poss(in_projdata_info_cyl_ptr->get_num_tangential_poss() -
            num_tang_poss_to_trim);
    
    shared_ptr<ProjDataInfo> out_projdata_info_ptr(out_projdata_info_cyl_ptr);
    ProjDataInterfile out_projdata(in_projdata_ptr->get_exam_info_sptr(), 
            out_projdata_info_ptr, output_filename, std::ios::out); 
    
    for (int seg = out_projdata.get_min_segment_num(); 
             seg <= out_projdata.get_max_segment_num();
             ++seg)
    {
      // keep sinograms out of the loop to avoid reallocations
      // initialise to something because there's no default constructor
      Sinogram<float> out_sino = 
        out_projdata.get_empty_sinogram(out_projdata.get_min_axial_pos_num(seg),seg);
      Sinogram<float> in_sino = 
        in_projdata_ptr->get_empty_sinogram(in_projdata_ptr->get_min_axial_pos_num(seg),seg);

      for (int ax = out_projdata.get_min_axial_pos_num(seg); 
               ax <= out_projdata.get_max_axial_pos_num(seg);
               ++ax )
      {
        out_sino= out_projdata.get_empty_sinogram(ax, seg);
        in_sino = in_projdata_ptr->get_sinogram(ax, seg);
        
        {
        for (int view=out_projdata.get_min_view_num();
                 view <= out_projdata.get_max_view_num();
                 ++view)
          for (int tang=out_projdata.get_min_tangential_pos_num();
                   tang <= out_projdata.get_max_tangential_pos_num();
                   ++tang)
            out_sino[view][tang] = in_sino[view][tang];
        }
        out_projdata.set_sinogram(out_sino);
      }
      
    }
      
  }
  else if (in_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry() ==
            "BlocksOnCylindrical")
  {
    const ProjDataInfoBlocksOnCylindrical * const in_projdata_info_blk_ptr =
      dynamic_cast<ProjDataInfoBlocksOnCylindrical const * >(in_projdata_ptr->get_proj_data_info_sptr());
    if (in_projdata_info_blk_ptr== NULL)
      {
        error("error converting to BlocksOnCylindrical projection data\n");
      }
    ProjDataInfoBlocksOnCylindrical * out_projdata_info_blk_ptr = 
      dynamic_cast<ProjDataInfoBlocksOnCylindrical * >
      (in_projdata_info_blk_ptr->clone());

    out_projdata_info_blk_ptr->
      set_num_tangential_poss(in_projdata_info_blk_ptr->get_num_tangential_poss() -
            num_tang_poss_to_trim);
    
    shared_ptr<ProjDataInfo> out_projdata_info_ptr(out_projdata_info_blk_ptr);
    ProjDataInterfile out_projdata(in_projdata_ptr->get_exam_info_sptr(), 
            out_projdata_info_ptr, output_filename, std::ios::out); 
            
  
    for (int seg = out_projdata.get_min_segment_num(); 
             seg <= out_projdata.get_max_segment_num();
             ++seg)
    {
      // keep sinograms out of the loop to avoid reallocations
      // initialise to something because there's no default constructor
      Sinogram<float> out_sino = 
        out_projdata.get_empty_sinogram(out_projdata.get_min_axial_pos_num(seg),seg);
      Sinogram<float> in_sino = 
        in_projdata_ptr->get_empty_sinogram(in_projdata_ptr->get_min_axial_pos_num(seg),seg);

      for (int ax = out_projdata.get_min_axial_pos_num(seg); 
               ax <= out_projdata.get_max_axial_pos_num(seg);
               ++ax )
      {
        out_sino= out_projdata.get_empty_sinogram(ax, seg);
        in_sino = in_projdata_ptr->get_sinogram(ax, seg);
        
        {
        for (int view=out_projdata.get_min_view_num();
                 view <= out_projdata.get_max_view_num();
                 ++view)
          for (int tang=out_projdata.get_min_tangential_pos_num();
                   tang <= out_projdata.get_max_tangential_pos_num();
                   ++tang)
            out_sino[view][tang] = in_sino[view][tang];
        }
        out_projdata.set_sinogram(out_sino);
      }
      
    }
      

  } 
  else if (in_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_scanner_geometry() ==
            "Generic")
  {
    const ProjDataInfoGeneric * const in_projdata_info_gen_ptr =
      dynamic_cast<ProjDataInfoGeneric const * >(in_projdata_ptr->get_proj_data_info_sptr());
    if (in_projdata_info_gen_ptr== NULL)
      {
        error("error converting to Generic projection data\n");
      }
    ProjDataInfoGeneric * out_projdata_info_gen_ptr = 
      dynamic_cast<ProjDataInfoGeneric * >
      (in_projdata_info_gen_ptr->clone());

    out_projdata_info_gen_ptr->
      set_num_tangential_poss(in_projdata_info_gen_ptr->get_num_tangential_poss() -
            num_tang_poss_to_trim);
    
    shared_ptr<ProjDataInfo> out_projdata_info_ptr(out_projdata_info_gen_ptr);
    ProjDataInterfile out_projdata(in_projdata_ptr->get_exam_info_sptr(), 
            out_projdata_info_ptr, output_filename, std::ios::out); 
            
    for (int seg = out_projdata.get_min_segment_num(); 
             seg <= out_projdata.get_max_segment_num();
             ++seg)
    {
      // keep sinograms out of the loop to avoid reallocations
      // initialise to something because there's no default constructor
      Sinogram<float> out_sino = 
        out_projdata.get_empty_sinogram(out_projdata.get_min_axial_pos_num(seg),seg);
      Sinogram<float> in_sino = 
        in_projdata_ptr->get_empty_sinogram(in_projdata_ptr->get_min_axial_pos_num(seg),seg);

      for (int ax = out_projdata.get_min_axial_pos_num(seg); 
               ax <= out_projdata.get_max_axial_pos_num(seg);
               ++ax )
      {
        out_sino= out_projdata.get_empty_sinogram(ax, seg);
        in_sino = in_projdata_ptr->get_sinogram(ax, seg);
        
        {
        for (int view=out_projdata.get_min_view_num();
                 view <= out_projdata.get_max_view_num();
                 ++view)
          for (int tang=out_projdata.get_min_tangential_pos_num();
                   tang <= out_projdata.get_max_tangential_pos_num();
                   ++tang)
            out_sino[view][tang] = in_sino[view][tang];
        }
        out_projdata.set_sinogram(out_sino);
      }
      
    }
  
  }
  else
  {
    error("error the scanner geometry of projection data is not known\n");
  }

  return EXIT_SUCCESS;
}
