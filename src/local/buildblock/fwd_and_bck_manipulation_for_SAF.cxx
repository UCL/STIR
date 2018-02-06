
#include "local/stir/fwd_and_bck_manipulation_for_SAF.h"



START_NAMESPACE_STIR
void
find_inverse_and_bck_densels(DiscretisedDensity<3,float>& image,
			     VectorWithOffset<SegmentByView<float> *>& all_segments,
			     VectorWithOffset<SegmentByView<float> *>& attenuation_segmnets,			    
			     const int min_z, const int max_z,
			     const int min_y, const int max_y,
			     const int min_x, const int max_x,
			     ProjMatrixByDensel& proj_matrix, bool do_attenuation, 
			     const float threshold,bool normalize_result)				
{
 /* VectorWithOffset<SegmentByView<float> *> all_ones(0);
  all_ones[0] = 
    new SegmentByView<float>(all_segments[0]->get_empty_segment_by_view(segment_num));		 
  (*all_ones[0]).fill(1);*/
	
  ProjMatrixElemsForOneDensel probs;
  for (int z = min_z; z<= max_z; ++z)
  {
    for (int y = min_y; y<= max_y; ++y)
    {
      for (int x = min_x; x<= max_x; ++x)
      {
	Densel densel(z,y,x);
	proj_matrix.get_proj_matrix_elems_for_one_densel(probs, densel);
	
	float denominator = 0;
	float sensitivity= 0;
	for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
	element_ptr != probs.end();++element_ptr)
	{  
	  //cerr << element_ptr->segment_num() << endl;
	  const float val=element_ptr->get_value();	  
	  
	 if (element_ptr->axial_pos_num()<= all_segments[element_ptr->segment_num()]->get_max_axial_pos_num() &&
         element_ptr->axial_pos_num()>= all_segments[element_ptr->segment_num()]->get_min_axial_pos_num())
	 {      
	  const float bin= 
 	  max(threshold,(*all_segments[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()]);
       	  denominator+= square(val);	  

	  if (do_attenuation)
	  {
	    float bin_attenuation= 
	      (*attenuation_segmnets[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()];	  	    
	    image[z][y][x] += (bin_attenuation/bin) * square(val);
	    sensitivity+=bin_attenuation*val;
	  }
	  else
	  {
	    image[z][y][x] += (1.F/bin) * square(val);
	    sensitivity+=val;
	  }
	 }
	}
	image[z][y][x]/= square(sensitivity);
	if (normalize_result)      
	  image[z][y][x]/= denominator;

      }
      
    }
  }
  

 
  for (DiscretisedDensity<3,float>::full_iterator iter = image.begin_all();
  iter !=image.end_all();
  ++iter)
    *iter = sqrt(*iter);
}


void
do_segments_densels_fwd(const VoxelsOnCartesianGrid<float>& image, 
            ProjData& proj_data,
	    VectorWithOffset<SegmentByView<float> *>& all_segments,
            const int min_z, const int max_z,
            const int min_y, const int max_y,
            const int min_x, const int max_x,
	    ProjMatrixByDensel& proj_matrix)
{
  
  ProjMatrixElemsForOneDensel probs;
  for (int z = min_z; z<= max_z; ++z)
  {
    for (int y = min_y; y<= max_y; ++y)
    {
      for (int x = min_x; x<= max_x; ++x)
      {
        if (image[z][y][x] == 0)
          continue;
        Densel densel(z,y,x);
        proj_matrix.get_proj_matrix_elems_for_one_densel(probs, densel);

        for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
	element_ptr != probs.end();
	++element_ptr)
        {
          if (element_ptr->axial_pos_num()<= proj_data.get_max_axial_pos_num(element_ptr->segment_num()) &&
	    element_ptr->axial_pos_num()>= proj_data.get_min_axial_pos_num(element_ptr->segment_num()))
            (*all_segments[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()] +=
	    image[z][y][x] * element_ptr->get_value();
        }
      }
    }
  }
  
}

void
fwd_densels_all(VectorWithOffset<SegmentByView<float> *>& all_segments, 
		shared_ptr<ProjMatrixByDensel> proj_matrix_ptr, 
		shared_ptr<ProjData > proj_data_ptr,
		const int min_z, const int max_z,
		const int min_y, const int max_y,
		const int min_x, const int max_x,
		const DiscretisedDensity<3,float>& in_density)
		
{
  
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  
  do_segments_densels_fwd(in_density_cast_0, 
			  *proj_data_ptr,
			   all_segments,
			   min_z, max_z,
			   min_y, max_y,
			   min_x, max_x,
			   *proj_matrix_ptr);  
  
}

END_NAMESPACE_STIR