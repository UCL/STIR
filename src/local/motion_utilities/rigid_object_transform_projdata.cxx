//
// $Id$
//
/*!

  \file
  \ingroup utilities
  \brief A utility to rorate projection data along the axial direction

  This can be used as a crude way for motion correction, when the motion is only in 
  z-direction.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/round.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/Quaternion.h"
#include "stir/CPUTimer.h"
#include <string>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
using std::min;
using std::max;
#endif

USING_NAMESPACE_STIR

class TF
{
public:
  TF(const shared_ptr<ProjDataInfo>& out_proj_data_info_ptr,
     const shared_ptr<ProjDataInfo>& in_proj_data_info_ptr,
     const RigidObject3DTransformation& transformation)
    : out_proj_data_info_ptr(out_proj_data_info_ptr),
      in_proj_data_info_ptr(in_proj_data_info_ptr),
      transformation(transformation)
  {
     out_proj_data_info_noarccor_ptr = 
       dynamic_cast<ProjDataInfoCylindricalNoArcCorr*>(out_proj_data_info_ptr.get());
     in_proj_data_info_noarccor_ptr = 
       dynamic_cast<ProjDataInfoCylindricalNoArcCorr*>(in_proj_data_info_ptr.get());
     if (out_proj_data_info_noarccor_ptr == 0 ||
	 in_proj_data_info_noarccor_ptr == 0)
       error("Wrong type of proj_data_info\n");


  }

  void transform_bin(Bin& bin) const
  {
     transformation.transform_bin(bin, 
                                  *out_proj_data_info_noarccor_ptr,
		                  *in_proj_data_info_noarccor_ptr);
  }
	
private:
  shared_ptr<ProjDataInfo> out_proj_data_info_ptr;
  shared_ptr<ProjDataInfo> in_proj_data_info_ptr;
  ProjDataInfoCylindricalNoArcCorr *out_proj_data_info_noarccor_ptr;
  ProjDataInfoCylindricalNoArcCorr *in_proj_data_info_noarccor_ptr;
  RigidObject3DTransformation transformation;
};

int main(int argc, char **argv)
{
  if (argc < 10 || argc > 12)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name q0 qx qy qz tx ty tz [max_in_segment_num_to_process [max_in_segment_num_to_process ]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  //const float angle_around_x =  atof(argv[3]) *_PI/180;
  const Quaternion<float> quat(atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]));
  const CartesianCoordinate3D<float> translation(atof(argv[9]),atof(argv[8]),atof(argv[7]));
  const int max_in_segment_num_to_process = argc <=10 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[10]);
  const int max_out_segment_num_to_process = argc <=11 ? max_in_segment_num_to_process : atoi(argv[11]);


  ProjDataInfo * proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
  proj_data_info_ptr->reduce_segment_range(-max_out_segment_num_to_process,max_out_segment_num_to_process);

  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  TF move_lor(out_projdata.get_proj_data_info_ptr()->clone(),
	      in_projdata_ptr->get_proj_data_info_ptr()->clone(),
	      RigidObject3DTransformation(quat, translation));
  const int out_min_segment_num = out_projdata.get_min_segment_num();
  const int out_max_segment_num = out_projdata.get_max_segment_num();
  VectorWithOffset<shared_ptr<SegmentByView<float> > > out_seg_ptr(out_min_segment_num, out_max_segment_num);
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    out_seg_ptr[segment_num] = 
      new SegmentByView<float>(out_projdata.get_empty_segment_by_view(segment_num));

  CPUTimer timer;
  timer.start();
  for (int segment_num = -max_in_segment_num_to_process;
       segment_num <= max_in_segment_num_to_process;
       ++segment_num)    
    {       
      const SegmentByView<float> in_segment = 
        in_projdata_ptr->get_segment_by_view( segment_num);
      cerr << "segment_num "<< segment_num << endl;
      const int in_max_ax_pos_num = in_segment.get_max_axial_pos_num();
      const int in_min_ax_pos_num = in_segment.get_min_axial_pos_num();
      const int in_max_view_num = in_segment.get_max_view_num();
      const int in_min_view_num = in_segment.get_min_view_num();
      const int in_max_tang_pos_num = in_segment.get_max_tangential_pos_num();
      const int in_min_tang_pos_num = in_segment.get_min_tangential_pos_num();
      for (int view_num=in_min_view_num; view_num<=in_max_view_num; ++view_num)
	for (int ax_pos_num=in_min_ax_pos_num; ax_pos_num<=in_max_ax_pos_num; ++ax_pos_num)
	  for (int tang_pos_num=in_min_tang_pos_num; tang_pos_num<=in_max_tang_pos_num; ++tang_pos_num)
	    {
	      Bin bin(segment_num, view_num, ax_pos_num, tang_pos_num,
		      in_segment[view_num][ax_pos_num][tang_pos_num]);
	      if (bin.get_bin_value()<=0)
		continue;
	      move_lor.transform_bin(bin);
	      if (bin.get_bin_value()>0)
		(*out_seg_ptr[bin.segment_num()])[bin.view_num()]
						 [bin.axial_pos_num()]
						 [bin.tangential_pos_num()] +=
		  bin.get_bin_value();
	    }
    }

  timer.stop();
  cerr << "CPU time " << timer.value() << endl;

  Succeeded succes = Succeeded::yes;
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    {       
      if (out_projdata.set_segment(*out_seg_ptr[segment_num]) == Succeeded::no)
             succes = Succeeded::no;
    }

    return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
