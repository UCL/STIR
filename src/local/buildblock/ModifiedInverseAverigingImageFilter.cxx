//
// $Id: 
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ModifiedInverseAverigingImageFilter
  
    \author Sanida Mustafovic
    \author Kris Thielemans
    
      \date $Date:
      \version $Revision:
*/
#include "local/tomo/ModifiedInverseAverigingImageFilter.h"
#include "IndexRange3D.h"
#include "shared_ptr.h"
#include "ProjData.h"
#include "VoxelsOnCartesianGrid.h"
#include "recon_buildblock/ForwardProjectorByBin.h"
#include "recon_buildblock/BackProjectorByBin.h"
#include "ProjDataFromStream.h"
#include "recon_buildblock/ProjMatrixByBin.h"
#include "local/tomo/recon_buildblock/ProjMatrixByDensel.h"
#include "local/tomo/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "interfile.h"
#include "CartesianCoordinate3D.h"

#include "local/tomo/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"

#include "local/recon_buildblock/BackProjectorByBinUsingSquareProjMatrixByBin.h"
#include "SegmentByView.h"

#include <iostream>
#include <fstream>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
using std::ios;
using std::find;
using std::iostream;
using std::fstream;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

  
// old stuff before densels
#if 1
// the following functions were used locally for kappa estimations.
void 
find_inverse(ProjDataFromStream*  proj_data_ptr_out,ProjDataFromStream * proj_data_ptr_in);

void 
find_inverse(VectorWithOffset<SegmentByView<float> *>& all_segments_inv,VectorWithOffset<SegmentByView<float> *>& all_segments_in);

void 
do_segments_fwd(const VoxelsOnCartesianGrid<float>& image, ProjData& s3d,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,	    	    
	    ForwardProjectorByBin&);
void
do_segments_bck(DiscretisedDensity<3,float>& image, 
		//shared_ptr<ProjDataFromStream>& proj_data_ptr,
		ProjData& proj_data_org,
		const int start_segment_num, const int end_segment_num,
		const int start_axial_pos_num, const int end_axial_pos_num,		
		const int start_view, const int end_view,
		const int start_tang_pos_num,const int end_tang_pos_num,
		BackProjectorByBin* back_projector_ptr,bool fill_with_1);
void
fwd_project(ProjData& proj_data,VoxelsOnCartesianGrid<float>* vox_image_ptr,
		const int start_segment_num, const int end_segment_num,
		const int start_axial_pos_num, const int end_axial_pos_num,		
		const int start_view, const int end_view,
		const int start_tang_pos_num,const int end_tang_pos_num);

void 
fwd_inverse_bck_individual_pixels(shared_ptr<ProjDataFromStream> proj_data_ptr,
			      VoxelsOnCartesianGrid<float>* vox_image_ptr_bck,
			      //VoxelsOnCartesianGrid<float>* vox_image_ptr,
			      const int start_segment_num, const int end_segment_num,
			      const int start_axial_pos_num, const int end_axial_pos_num,		
			      const int start_view, const int end_view,
			      const int start_tang_pos_num,const int end_tang_pos_num,
			      const DiscretisedDensity<3,float>& in_density);

void 
fwd_densels_invert_and_bck_individually(shared_ptr<ProjData > proj_data_ptr,
					shared_ptr<ProjMatrixByDensel> proj_matrix_ptr,
					//shared_ptr<ProjDataFromStream> proj_data_ptr,
	    VoxelsOnCartesianGrid<float>* vox_image_ptr,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,		
	    const int start_view, const int end_view,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const DiscretisedDensity<3,float>& in_density);


#endif


#if 1



void 
find_inverse(VectorWithOffset<SegmentByView<float> *>& all_segments_inv,VectorWithOffset<SegmentByView<float> *>& all_segments_in);

void
do_segments_densels_fwd(const VoxelsOnCartesianGrid<float>& image, 
            ProjData& proj_data,
	    VectorWithOffset<SegmentByView<float> *>& all_segments,
            const int min_z, const int max_z,
            const int min_y, const int max_y,
            const int min_x, const int max_x,
	    ProjMatrixByDensel& proj_matrix);


void
fwd_densels_all(VectorWithOffset<SegmentByView<float> *>& all_segments,
		shared_ptr<ProjMatrixByDensel> proj_matrix_ptr, 
		shared_ptr<ProjData > proj_data_ptr,
		const int min_z, const int max_z,
		const int min_y, const int max_y,
		const int min_x, const int max_x,
		const DiscretisedDensity<3,float>& in_density);

void
find_inverse_and_bck_densels(DiscretisedDensity<3,float>& image,
			     VectorWithOffset<SegmentByView<float> *>& all_segments,
			     VectorWithOffset<SegmentByView<float> *>& attenuation_segmnets,			
				const int min_z, const int max_z,
				const int min_y, const int max_y,
				const int min_x, const int max_x,
				ProjMatrixByDensel& proj_matrix, 
				bool do_attenuation,
				const float threshold);

void
find_inverse_and_bck_densels(DiscretisedDensity<3,float>& image,
			     VectorWithOffset<SegmentByView<float> *>& all_segments,
			     VectorWithOffset<SegmentByView<float> *>& attenuation_segmnets,			    
			     const int min_z, const int max_z,
			     const int min_y, const int max_y,
			     const int min_x, const int max_x,
			     ProjMatrixByDensel& proj_matrix, bool do_attenuation, 
			     const float threshold)				
{
  
    
  ProjMatrixElemsForOneDensel probs;
  for (int z = min_z; z<= max_z; ++z)
  {
    for (int y = min_y; y<= max_y; ++y)
    {
      for (int x = min_x; x<= max_x; ++x)
      {
	Densel densel(z,y,x);
	proj_matrix.get_proj_matrix_elems_for_one_densel(probs, densel);
	
	for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
	element_ptr != probs.end();++element_ptr)
	{
	  const float val=element_ptr->get_value();
	  
	  float bin= 
	    (*all_segments[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()];
	  if (bin >= threshold)
	  {
	    if (do_attenuation)
	    {
	      float bin_attenuation= 
		(*attenuation_segmnets[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()];	  
	      
	      image[z][y][x] += (bin_attenuation/bin) * square(val);}
	    else
	      image[z][y][x] += (1.F/bin) * square(val);
	  }
	  else
	    if(do_attenuation)
	    {
	      float bin_attenuation= 
		(*attenuation_segmnets[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()];	  
	      image[z][y][x] += (bin_attenuation/threshold) * square(val);
	    }
	    else
	      image[z][y][x] += (1.F/threshold) * square(val);
	    
	}
	
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


template <typename elemT>
ModifiedInverseAverigingImageFilter<elemT>::
ModifiedInverseAverigingImageFilter()
{ 
  set_defaults();
}


template <typename elemT>
ModifiedInverseAverigingImageFilter<elemT>::
ModifiedInverseAverigingImageFilter(string proj_data_filename_v,
				    string attenuation_proj_data_filename_v,
				    const VectorWithOffset<elemT>& filter_coefficients_v,
				    shared_ptr<ProjData> proj_data_ptr_v,
				    shared_ptr<ProjData> attenuation_proj_data_ptr_v,
				    int mask_size_v)

				    
{
  assert(filter_coefficients.get_length() == 0 ||
         filter_coefficients.begin()==0);
  
  for (int i = filter_coefficients_v.get_min_index();i<=filter_coefficients_v.get_max_index();i++)
    filter_coefficients[i] = filter_coefficients_v[i];
  proj_data_filename  = proj_data_filename_v;
  attenuation_proj_data_filename = attenuation_proj_data_filename_v;
  proj_data_ptr = proj_data_ptr_v;
  attenuation_proj_data_ptr = attenuation_proj_data_ptr_v;
  mask_size= mask_size_v;
}


template <typename elemT>
Succeeded 
ModifiedInverseAverigingImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
    proj_data_ptr = 
       ProjData::read_from_file( proj_data_filename); 
    
    if (attenuation_proj_data_filename !="1")
    attenuation_proj_data_ptr =
    ProjData::read_from_file(attenuation_proj_data_filename); 
	else 
    attenuation_proj_data_ptr = NULL;

   
    return Succeeded::yes;
  
}


// densel stuff - > apply
#if 1

template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const
{
  //the first time virtual_apply is called for this object, counter is set to 0
  static int count=0;
  // every time it's called, counter is incremented
  count++;
  cerr << " checking the counter  " << count << endl; 
  
  
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ModifiedInverseAverigingArrayFilter <3,float> >  > > > all_filter_coefficients;
  
  if (count==1)
  {
    all_filter_coefficients.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());
    
    for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
    {
      all_filter_coefficients[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
      for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
        (all_filter_coefficients[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
    }
    
  }
  
  if ( (count % 5) ==0 || count == 1 ) 
  {
    
    shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
    
    int limit_segments= 0;
    new_data_info_ptr->reduce_segment_range(-limit_segments, limit_segments);
    
    
    VoxelsOnCartesianGrid<float> *  vox_image_ptr_1 =
      new VoxelsOnCartesianGrid<float> (IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
      in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
      in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
      in_density.get_origin(),in_density_cast_0.get_voxel_size());  
    
    int start_segment_num = proj_data_ptr->get_min_segment_num();
    int end_segment_num = proj_data_ptr->get_max_segment_num();
    
    VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);
    VectorWithOffset<SegmentByView<float> *> all_attenuation_segments(start_segment_num, end_segment_num);

    // first initialise to false
    bool do_attenuation = false;
    
    for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    {
		all_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));

		if (attenuation_proj_data_filename !="1")
		{
		  do_attenuation = true;
		  all_attenuation_segments[segment_num] = 
		    new SegmentByView<float>(attenuation_proj_data_ptr->get_segment_by_view(segment_num));
		 }
		else 
		{
		do_attenuation = false;
		all_attenuation_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));		 
		(*all_attenuation_segments[segment_num]).fill(1);
		}
		
    }
    
    
    
    VectorWithOffset<SegmentByView<float> *> all_segments_for_kappa0(start_segment_num, end_segment_num);
    
    
    for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
      all_segments_for_kappa0[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    
    
    vox_image_ptr_1->set_origin(Coordinate3D<float>(0,0,0));   
    
    shared_ptr<DiscretisedDensity<3,float> > image_sptr =  vox_image_ptr_1;
    
    shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
      new ProjMatrixByDenselUsingRayTracing;
    
    proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
      image_sptr);
    cerr << proj_matrix_ptr->parameter_info();
    
    fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
      in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
      in_density_cast_0.get_min_y(), in_density_cast_0.get_max_y(),
      in_density_cast_0.get_min_x(), in_density_cast_0.get_max_x(),
      in_density_cast_0);
    
    VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa0 =
      new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
      in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
      in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
      in_density.get_origin(),in_density_cast_0.get_voxel_size());  
    
    
    shared_ptr<DiscretisedDensity<3,float> > kappa0_ptr_bck =  vox_image_ptr_kappa0;   
    
    VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa1 =
      new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
      in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
      in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
      in_density.get_origin(),in_density_cast_0.get_voxel_size());  
    
    shared_ptr<DiscretisedDensity<3,float> > kappa1_ptr_bck =  vox_image_ptr_kappa1;   

 // WARNING - find a way of finding max in the sinogram
  // TODO - include other segments as well
  const float max_in_viewgram = 33.52;
    //(proj_data_ptr->get_segment_by_view(0)).find_max();
  //cerr <<  max_in_viewgram ;
  //cerr << endl;
  //33.52F;
  const float threshold = 0.0001F*max_in_viewgram;  

  cerr << " THRESHOLD IS" << threshold; 
  cerr << endl;
    
    find_inverse_and_bck_densels(*kappa1_ptr_bck,all_segments,
      all_attenuation_segments,
      vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
      vox_image_ptr_kappa1->get_min_y(),vox_image_ptr_kappa1->get_max_y(),
      vox_image_ptr_kappa1->get_min_x(),vox_image_ptr_kappa1->get_max_x(),
      *proj_matrix_ptr, do_attenuation,threshold);
    
    for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    { 
      delete all_segments[segment_num];
    }   
    
    cerr << "min and max in image - kappa1 " <<kappa1_ptr_bck->find_min()
      << ", " << kappa1_ptr_bck->find_max() << endl;   
    
    char* file1 = "kappa1";
    //cerr <<"  - Saving " << file1 << endl;
    write_basic_interfile(file1, *kappa1_ptr_bck);
    
    
    const string filename ="kapa0_div_kapa1_pf";
    shared_ptr<iostream> output = 
      new fstream (filename.c_str(), ios::trunc|ios::in|ios::out|ios::binary);
    
    const string filename1 ="values_of_kapa0_and_kapa1_pf";
    shared_ptr<iostream> output1 =     
      new fstream (filename1.c_str(), ios::trunc|ios::in|ios::out|ios::binary);
    
    
    if (!*output1)
      error("Error opening output file %s\n",filename1.c_str());
    
    if (!*output)
      error("Error opening output file %s\n",filename.c_str());
    
    
    *output << "kapa0_div_kapa1" << endl;
    *output << endl;
    *output << endl;
    *output << "Plane number " << endl;   
    
    int size = filter_coefficients.get_length();
    
    
    for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
     for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
      for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
      	{
	  
	  // WARNING - only works for segment zero at the moment
	  // do the calculation of kappa0 here
	  kappa0_ptr_bck->fill(0); 
	  (*all_segments_for_kappa0[all_segments.get_min_index()]).fill(0);	    
	  if (attenuation_proj_data_filename !="1")
	  {
	   
	    shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	      new VoxelsOnCartesianGrid<float>
	      (IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
	      -mask_size,mask_size,
	      -mask_size,mask_size),in_density.get_origin(),in_density_cast_0.get_voxel_size());  
	   
	    const int min_j = max(in_density_cast_0.get_min_y(),j-mask_size);
	    const int max_j = min(in_density_cast_0.get_max_y(),j+mask_size);
	    const int min_i = max(in_density_cast_0.get_min_x(),i-mask_size);
	    const int max_i = min(in_density_cast_0.get_max_x(),i+mask_size);
	   
	    // the mask size is in 2D only
	    for (int j_in =min_j;j_in<=max_j;j_in++)
	      for (int i_in =min_i;i_in<=max_i;i_in++)	
	     
		    (*in_density_cast_tmp)[k][j_in-j][i_in-i] = in_density_cast_0[k][j_in][i_in];
	      
	      fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
		in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
		in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
		in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
		*in_density_cast_tmp);
	      
	      find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
		all_attenuation_segments,
		vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
		0,0,0,0,
		*proj_matrix_ptr,false,threshold);	  
	      (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[k][0][0];
	      
	      
	  }
	  else
	  {
	    const int min_j = max(in_density_cast_0.get_min_y(),j-mask_size);
	    const int max_j = min(in_density_cast_0.get_max_y(),j+mask_size);
	    const int min_i = max(in_density_cast_0.get_min_x(),i-mask_size);
	    const int max_i = min(in_density_cast_0.get_max_x(),i+mask_size);
	    
	    fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	      in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
	      min_j,max_j,
	      min_i,max_i,
	      //j-2,j+2,
	      //i-2,i+2,
	      in_density_cast_0);
	    
	    find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	      all_attenuation_segments,
	      vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
	      j,j,i,i,
	      *proj_matrix_ptr,false,threshold);
	  }
	  //	cerr << "min and max in image - kappa0 " <<kappa0_ptr_bck->find_min()
	  //	<< ", " << kappa0_ptr_bck->find_max() << endl; 
	  
	  char* file0 = "kappa0";
	  write_basic_interfile(file0, *kappa0_ptr_bck);
	  
	  float sq_kapas;
	  
	  if ( fabs((double)(*kappa1_ptr_bck)[k][j][i]) > 0.00000000000001 && 
	    fabs((double)(*kappa0_ptr_bck)[k][j][i]) > 0.00000000000001 )
	  { 
	    sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);
	    
	    *output1 << " Values of kapa0 and kapa1" << endl;
	    *output1<< "for k   "<< k;
	    *output1 <<":";
	    *output1 << j;
	    *output1 <<",";
	    *output1 <<i;
	    *output1 <<"    ";
	    //*output1 <<(*image_sptr_0)[k][j][i];
	    *output1 <<(*kappa0_ptr_bck)[k][j][i];
	    *output1 << "     ";
	    *output1 <<(*kappa1_ptr_bck)[k][j][i];
	    *output1 << endl;
	    *output<< "for k   "<< k;
	    *output <<":";
	    *output << j;
	    *output <<",";
	    *output <<i;
	    *output <<"    ";
	    *output << sq_kapas;
	    *output <<endl;
	    
	    inverse_filter = 
	      ModifiedInverseAverigingArrayFilter<3,elemT>(filter_coefficients,sq_kapas);	  
	    
	    all_filter_coefficients[k][j][i] = 
	      new ModifiedInverseAverigingArrayFilter<3,elemT>(inverse_filter);		
	    
	  }
	  else
	  {	
	    sq_kapas = 0;
	    inverse_filter = 
	      ModifiedInverseAverigingArrayFilter<3,elemT>();	  
	    all_filter_coefficients[k][j][i] =
	      new ModifiedInverseAverigingArrayFilter<3,elemT>(inverse_filter);
	    
	  }
	  
	}      
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	{ 
	  delete all_segments_for_kappa0[segment_num];
	  delete all_attenuation_segments[segment_num];
	}   
     }
     
     for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
       for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
	 for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
 	 {
	   Array<3,elemT> tmp_out(IndexRange3D(k,k,j,j,i,i));
	   (*all_filter_coefficients[k][j][i])(tmp_out,in_density);
	   out_density[k][j][i] = tmp_out[k][j][i];	
	   
	   
	 }
	 
}

/*const IndexRange<3> kernel_index_range =
	   all_filter_coefficients[k][j][i]->get_kernel_index_range();
	   IndexRange<3> index_range;
	   {
	   int k_in_min = max(in_density.get_min_index()-k,kernel_index_range.get_min_index());
	   int k_in_max = min(in_density.get_max_index()-k,kernel_index_range.get_max_index());
	   index_range.grow(k_in_min, k_in_max);
	   for (int k_in=k_in_min; k_in<=k_in_max;k_in++)   
	   {
	   int j_in_min =max(in_density[k_in+k].get_min_index()-j,kernel_index_range[k_in].get_min_index());
	   int j_in_max =min(in_density[k_in+k].get_max_index()-j,kernel_index_range[k_in].get_max_index());	      
	   index_range[k_in].grow(j_in_min, j_in_max);
	   for (int j_in=j_in_min; j_in<=j_in_max;j_in++)   
	   {
	   int i_in_min =max(in_density[k_in+k][j_in+j].get_min_index()-i,kernel_index_range[k_in][j_in].get_min_index());
	   int i_in_max =min(in_density[k_in+k][j_in+j].get_max_index()-i,kernel_index_range[k_in][j_in].get_max_index());
	   index_range[k_in][j_in] = IndexRange<1>(i_in_min, i_in_max);
	   }
	   }
	   }  
	   Array<3,elemT> tmp_out(index_range);
	   for (int k_in=index_range.get_min_index();
	   k_in<=index_range.get_max_index();
	   k_in++)   
	   for (int j_in =index_range[k_in].get_min_index();
	   j_in<=index_range[k_in].get_max_index();
	   j_in++)
	   for (int i_in =index_range[k_in][j_in].get_min_index();
	   i_in<=index_range[k_in][j_in].get_max_index();
	   i_in++)	
	   {
	   tmp_out[k_in][j_in][i_in] = in_density[k_in+k][j_in+j][i_in+i];
	   }
	   (*all_filter_coefficients[k][j][i])(tmp_out);	  
	   out_density[k][j][i] = tmp_out[0][0][0];*/
	   
	   

#endif


// OLD
# if 1


void 
fwd_inverse_bck_individual_pixels(shared_ptr<ProjDataFromStream> proj_data_ptr,
			      VoxelsOnCartesianGrid<float>* vox_image_ptr_bck,
			      //VoxelsOnCartesianGrid<float>* vox_image_ptr,
			      const int start_segment_num, const int end_segment_num,
			      const int start_axial_pos_num, const int end_axial_pos_num,		
			      const int start_view, const int end_view,
			      const int start_tang_pos_num,const int end_tang_pos_num,
			      const DiscretisedDensity<3,float>& in_density)
{
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
   VoxelsOnCartesianGrid<float> *  in_density_ptr =
     in_density_cast_0.get_empty_voxels_on_cartesian_grid();

  const float z_origin = 0;
  
  vox_image_ptr_bck->set_origin(Coordinate3D<float>(z_origin,0,0));  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr =  in_density_ptr;
    //vox_image_ptr_bck;
  // in_density_ptr;
 
  string name = "Ray Tracing";
  
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    ForwardProjectorByBin::read_registered_object(0, name);
  
  forw_projector_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
    image_sptr);
  
  cerr << forw_projector_ptr->parameter_info();
  list<ViewSegmentNumbers> already_processed;  
  
 
  VoxelsOnCartesianGrid<float> * in_density_extracted_ptr =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
    //line source + cylinder filtered
    -26,26,-13,13),
    //-10,10,-10,10),
    in_density.get_origin(),in_density_cast_0.get_voxel_size());  
  
    in_density_extracted_ptr ->fill(0);
  
  in_density_extracted_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));    
  
  //int counter;
  for ( int k = in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k+=1)
    // to do all pixels where the cyl and the line source are
    for ( int j =-3;j<=3;j++)
     for ( int i =-3;i<=3;i++) 
      {
	in_density_extracted_ptr->fill(0);	
	for (int mask_j = -2; mask_j<=2; mask_j++)
	  for (int mask_i = -2; mask_i<=2; mask_i++)	 
	  {
	    (*in_density_extracted_ptr)[k][j+mask_j][i+mask_i] = in_density[k][j+mask_j][i+mask_i];      	  
	  }
	  
	  // do the fwd
	  if (proj_data_ptr->get_min_tangential_pos_num()==start_tang_pos_num && 
	    proj_data_ptr->get_min_view_num() == start_view && 
	    proj_data_ptr->get_min_axial_pos_num(0)== start_axial_pos_num)
	  {  	    
	    do_segments_fwd(*in_density_extracted_ptr,(*proj_data_ptr), 
	      start_segment_num,  end_segment_num,	
	      start_axial_pos_num,end_axial_pos_num,
	      start_view, end_view,
	      start_tang_pos_num, end_tang_pos_num,
	      *forw_projector_ptr);  	    
	  }
	  else
	  {    
	    // first set all data to 0
	    cerr << "Filling output file with 0\n";
	    for (int segment_num = proj_data_ptr->get_min_segment_num(); 
	    segment_num <= proj_data_ptr->get_max_segment_num(); 
	    ++segment_num)
	    {
	      const SegmentByView<float> segment = 
		proj_data_ptr->get_proj_data_info_ptr()->get_empty_segment_by_view(segment_num, false);
	      if (!(proj_data_ptr->set_segment(segment) == Succeeded::yes))
		warning("Error set_segment %d\n", segment_num);            
	    }
	    do_segments_fwd(*in_density_extracted_ptr,(*proj_data_ptr), 
	      start_segment_num,  end_segment_num,	
	      start_axial_pos_num,end_axial_pos_num,
	      start_view, end_view,
	      -13,13,
	      *forw_projector_ptr);   
	  }	  
	  
	  // now find the inverse
	  shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
	  
	  int limit_segments= 0;
	  new_data_info_ptr->reduce_segment_range(-limit_segments, limit_segments);
	  
	  const string output_file_name_kappa0 = "fwd_kappa0.s";
	  
	  shared_ptr<iostream> sino_stream_kappa0 = 
	    new fstream (output_file_name_kappa0.c_str(),ios::trunc|ios::out|ios::in|ios::binary);
	  
	  
	  if (!sino_stream_kappa0->good())
	  {
	    error("SV filters: error opening files %s\n",output_file_name_kappa0.c_str());
	  }
	  
	  //DiscretisedDensity<3,float>* in_density_tmp_0 =  in_density.clone();  
	  
	  //VoxelsOnCartesianGrid<float> * vox_image_ptr_1 =
	    //new VoxelsOnCartesianGrid<float>(*new_data_info_ptr);   
	  
	  shared_ptr<ProjDataFromStream> proj_data_ptr_kappa0 =
	    new ProjDataFromStream(new_data_info_ptr,sino_stream_kappa0);
	  
	  
	  // after the kappas are obtained -> find the inverse 
	  
	  const string output_file_name_inverse_kappa0 = "inv_kappa0.s";
	  
	  shared_ptr<iostream> sino_stream_inv_kappa0 = 
	    new fstream (output_file_name_inverse_kappa0.c_str(),ios::trunc| ios::out|ios::in|ios::binary);
	  if (!sino_stream_inv_kappa0->good())
	  {
	    error("SV filters: error opening files %s\n",output_file_name_inverse_kappa0.c_str());
	  }
	  
	  shared_ptr<ProjDataFromStream> proj_data_ptr_inv_kappa0 =
	    new ProjDataFromStream(new_data_info_ptr,sino_stream_inv_kappa0);	  	  
	  find_inverse(proj_data_ptr_inv_kappa0.get(),proj_data_ptr.get());
	  
	  // now do the back_projection
	  
	  // sm 05/10/2001 - > use the same image_sptr as for fwd:

	  VoxelsOnCartesianGrid<float> * vox_image_ptr_kappa0 =
	    in_density_cast_0.get_empty_voxels_on_cartesian_grid();
	    /*new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
	    in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
	    in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
	    in_density.get_origin(),in_density_cast_0.get_voxel_size());  */
	  
	  //vox_image_ptr_kappa0 = &in_density_cast_0 ;
	  shared_ptr<DiscretisedDensity<3,float> > image_sptr_0 =  vox_image_ptr_kappa0;   
	  
	  // JUST TO CHECK IT -> BY HAND
	  const ProjDataInfo * proj_data_info_ptr = 
	    proj_data_ptr->get_proj_data_info_ptr();
	  
	  string name = "Ray Tracing";
	  shared_ptr<ProjMatrixByBin> PM_0 = 
	    ProjMatrixByBin::read_registered_object(0, name);
	  PM_0->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),image_sptr_0); 
	  BackProjectorByBin*  bck_projector_ptr_0  =
	    new BackProjectorByBinUsingSquareProjMatrixByBin(PM_0);
	  
	  bool fill_with_1 = false;
	  do_segments_bck(*image_sptr_0, 
	    *proj_data_ptr_inv_kappa0,   
	    proj_data_ptr->get_min_segment_num(), proj_data_ptr->get_max_segment_num(), 
	    proj_data_ptr->get_min_axial_pos_num(0),proj_data_ptr->get_max_axial_pos_num(0),
	    proj_data_ptr->get_min_view_num(),proj_data_ptr->get_max_view_num(),
	    proj_data_ptr->get_min_tangential_pos_num(), 
	    proj_data_ptr->get_max_tangential_pos_num(),
	    bck_projector_ptr_0, fill_with_1);    
	  (*vox_image_ptr_bck) [k][j][i] = (*image_sptr_0)[k][j][i];  
	  
	  //delete vox_image_ptr_1 ;
      delete bck_projector_ptr_0 ;
      }

    //  delete  in_density_ptr ;
      delete in_density_extracted_ptr;    
       for (VoxelsOnCartesianGrid<float>::full_iterator iter = vox_image_ptr_bck->begin_all();
         iter != vox_image_ptr_bck->end_all();
	 ++iter)
      *iter = sqrt(*iter);
     
     
}



void
fwd_project_individual_pixels(ProjData& proj_data,VoxelsOnCartesianGrid<float>* vox_image_ptr,
			      const int start_segment_num, const int end_segment_num,
			      const int start_axial_pos_num, const int end_axial_pos_num,		
			      const int start_view, const int end_view,
			      const int start_tang_pos_num,const int end_tang_pos_num,
			      const DiscretisedDensity<3,float>& in_density)
{
  
  // extract the neighbourhood of the current pixel that is 
  // being fwd projected
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  const float z_origin = 0;
  
  vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr = vox_image_ptr;
  
  // use shared_ptr such that it cleans up automatically
  string name = "Ray Tracing";
  
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    ForwardProjectorByBin::read_registered_object(0, name);
  
  forw_projector_ptr->set_up(proj_data.get_proj_data_info_ptr()->clone(),
    image_sptr);

  cerr << forw_projector_ptr->parameter_info();
  list<ViewSegmentNumbers> already_processed;  
 
  VoxelsOnCartesianGrid<float> * in_density_extracted_ptr =
    vox_image_ptr->get_empty_voxels_on_cartesian_grid();
  
  in_density_extracted_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));    
  
  //int counter;
  for ( int k = 40;k<=40;k+=20)
    for ( int j =-1;j<=1;j++)
      for ( int i =-1;i<=1;i++)     
     {
	in_density_extracted_ptr->fill(0);	
	for (int mask_j = -2; mask_j<=2; mask_j++)
	  for (int mask_i = -2; mask_i<=2; mask_i++)	 
	  {
	  (*in_density_extracted_ptr)[k][j+mask_j][i+mask_i] = in_density[k][j+mask_j][i+mask_i];      	  
	  }
    
  if (proj_data.get_min_tangential_pos_num()==start_tang_pos_num && 
	    proj_data.get_min_view_num() == start_view && 
	    proj_data.get_min_axial_pos_num(0)== start_axial_pos_num)
	  {  	    
	  do_segments_fwd(*in_density_extracted_ptr,proj_data, 
	    start_segment_num,  end_segment_num,	
	    start_axial_pos_num,end_axial_pos_num,
	    start_view, end_view,
	    start_tang_pos_num, end_tang_pos_num,
	    *forw_projector_ptr);  	    
	  }
	  else
	  {    
	    // first set all data to 0
	    cerr << "Filling output file with 0\n";
	    for (int segment_num = proj_data.get_min_segment_num(); 
	    segment_num <= proj_data.get_max_segment_num(); 
	    ++segment_num)
	    {
	      const SegmentByView<float> segment = 
		proj_data.get_proj_data_info_ptr()->get_empty_segment_by_view(segment_num, false);
	      if (!(proj_data.set_segment(segment) == Succeeded::yes))
		warning("Error set_segment %d\n", segment_num);            
	    }
	    do_segments_fwd(*in_density_extracted_ptr,proj_data, 
	    start_segment_num,  end_segment_num,	
	    start_axial_pos_num,end_axial_pos_num,
	    start_view, end_view,
	    //start_tang_pos_num, end_tang_pos_num,
	    -6,6,
	    *forw_projector_ptr);   
	  }	  
	//*vox_image_ptr= *in_density_extracted_ptr;
	//(*vox_image_ptr)[k][j][i]= (*in_density_extracted_ptr)[k][j][i];
	 
      }
      	
}

void
fwd_project_individual_pixels_cyl(ProjData& proj_data,VoxelsOnCartesianGrid<float>* vox_image_ptr,
			      const int start_segment_num, const int end_segment_num,
			      const int start_axial_pos_num, const int end_axial_pos_num,		
			      const int start_view, const int end_view,
			      const int start_tang_pos_num,const int end_tang_pos_num,
			      const DiscretisedDensity<3,float>& in_density)
{
  
  // extract the neighbourhood of the current pixel that is 
  // being fwd projected
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  const float z_origin = 0;
  
  vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));
  shared_ptr<DiscretisedDensity<3,float> > image_sptr = vox_image_ptr;
  
  // use shared_ptr such that it cleans up automatically
  string name = "Ray Tracing";
  
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    ForwardProjectorByBin::read_registered_object(0, name);
  // ask_type_and_parameters();
  
  forw_projector_ptr->set_up(proj_data.get_proj_data_info_ptr()->clone(),
    image_sptr);
  
  //cerr << forw_projector_ptr->parameter_info();
  list<ViewSegmentNumbers> already_processed;
  
  VoxelsOnCartesianGrid<float> * in_density_extracted_ptr =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
    -8,21,-8,21),
    in_density.get_origin(),in_density_cast_0.get_voxel_size());  
  
  for ( int k = 40;k<=in_density.get_max_index();k+=20)
    for ( int j = 11;j<=18;j++)
      for ( int i =11;i<=18;i++)
      {
	in_density_extracted_ptr->fill(0);
	for (int mask_j = -3; mask_j <= 3; mask_j++)
	  for (int mask_i = -3; mask_i <= 3; mask_i++)	
	{
	  (*in_density_extracted_ptr)[k][j+mask_j][i+mask_i] = in_density[k][j+mask_j][i+mask_i];      
	}
		
	// first set all data to 0
	//cerr << "Filling output file with 0\n";
	for (int segment_num = proj_data.get_min_segment_num(); 
	segment_num <= proj_data.get_max_segment_num(); 
	++segment_num)
	{
	  const SegmentByView<float> segment = 
	    proj_data.get_proj_data_info_ptr()->get_empty_segment_by_view(segment_num, false);
	  if (!(proj_data.set_segment(segment) == Succeeded::yes))
	    warning("Error set_segment %d\n", segment_num);            
	}
	
	for  (int i= start_tang_pos_num ; i<= end_tang_pos_num;i++)
	  
	  do_segments_fwd(*in_density_extracted_ptr,proj_data, 
	  start_segment_num,  end_segment_num,	
	  start_axial_pos_num,end_axial_pos_num,
	  start_view, end_view,
	  start_tang_pos_num, end_tang_pos_num,
	  *forw_projector_ptr);    
	  (*vox_image_ptr)[k][j][i] = (*in_density_extracted_ptr)[k][j][i];
      }


	
}

void
fwd_project(ProjData& proj_data,VoxelsOnCartesianGrid<float>* vox_image_ptr,
		const int start_segment_num, const int end_segment_num,	
		const int start_axial_pos_num, const int end_axial_pos_num,
		const int start_view, const int end_view,
		const int start_tang_pos_num,const int end_tang_pos_num)
{
   
  const float z_origin = 0;
   
  vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));
  shared_ptr<DiscretisedDensity<3,float> > image_sptr = vox_image_ptr;

  // use shared_ptr such that it cleans up automatically
  string name = "Ray Tracing";

  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    ForwardProjectorByBin::read_registered_object(0, name);
   // ask_type_and_parameters();

  forw_projector_ptr->set_up(proj_data.get_proj_data_info_ptr()->clone(),
			     image_sptr);
 
  cerr << forw_projector_ptr->parameter_info();
  list<ViewSegmentNumbers> already_processed;

  if (proj_data.get_min_tangential_pos_num()==start_tang_pos_num && 
    proj_data.get_min_view_num() == start_view && 
    proj_data.get_min_axial_pos_num(0)== start_axial_pos_num)
  {   
       
    do_segments_fwd(*vox_image_ptr, proj_data,
      start_segment_num,  end_segment_num,	
      start_axial_pos_num,end_axial_pos_num,
      start_view, end_view,start_tang_pos_num,end_tang_pos_num,
      *forw_projector_ptr);
    
  }
  else
  {    
    // first set all data to 0
    cerr << "Filling output file with 0\n";
    for (int segment_num = proj_data.get_min_segment_num(); 
         segment_num <= proj_data.get_max_segment_num(); 
         ++segment_num)
    {
      const SegmentByView<float> segment = 
        proj_data.get_proj_data_info_ptr()->get_empty_segment_by_view(segment_num, false);
      if (!(proj_data.set_segment(segment) == Succeeded::yes))
        warning("Error set_segment %d\n", segment_num);            
    }
   // do
   // {
      do_segments_fwd(*vox_image_ptr,proj_data, 
	          start_segment_num,  end_segment_num,	
		  start_axial_pos_num,end_axial_pos_num,
	          start_view, end_view,
                  start_tang_pos_num, end_tang_pos_num,
	          *forw_projector_ptr);      
  //  }
  }
   
}

void
do_segments_fwd(const VoxelsOnCartesianGrid<float>& image, ProjData& proj_data,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,	    
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin& forw_projector)
{
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    forw_projector.get_symmetries_used()->clone();  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)      
    {       
      ViewSegmentNumbers vs(view, segment_num);
      symmetries_sptr->find_basic_view_segment_numbers(vs);
      if (find(already_processed.begin(), already_processed.end(), vs)
        != already_processed.end())
        continue;
      
      already_processed.push_back(vs);
      
     /*cerr << "Processing view " << vs.view_num() 
        << " of segment " <<vs.segment_num()
        << endl;*/
      
      RelatedViewgrams<float> viewgrams = 
        proj_data.get_empty_related_viewgrams(vs, symmetries_sptr,false);
      forw_projector.forward_project(viewgrams, image,
        viewgrams.get_min_axial_pos_num(),
        viewgrams.get_max_axial_pos_num(),
        start_tangential_pos_num, end_tangential_pos_num);	  
        if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
        error("Error set_related_viewgrams\n");            
    }   
}

void 
find_inverse(VectorWithOffset<SegmentByView<float> *>& all_segments_inv,VectorWithOffset<SegmentByView<float> *>& all_segments_in)
{
  
  const float max_in_viewgram = 33.52F;
  const float threshold = 0.0001F*max_in_viewgram;    
  float inv;
  
  for ( int segment_num = all_segments_in.get_min_index(); segment_num <= all_segments_in.get_max_index();segment_num++)
    for ( int view_num =all_segments_in[all_segments_in.get_min_index()]->get_min_view_num(); view_num <= all_segments_in[all_segments_in.get_min_index()]->get_max_view_num(); view_num++)
      for ( int axial_pos_num = all_segments_in[all_segments_in.get_min_index()]->get_min_axial_pos_num(); axial_pos_num <= all_segments_in[all_segments_in.get_min_index()]->get_max_axial_pos_num();axial_pos_num++)
	for ( int tang_pos_num = all_segments_in[all_segments_in.get_min_index()]->get_min_tangential_pos_num(); tang_pos_num <=all_segments_in[all_segments_in.get_min_index()]->get_max_tangential_pos_num();tang_pos_num++)
	{	
	  float bin= (*all_segments_in[segment_num])[view_num][axial_pos_num][tang_pos_num] ;
	  
	  if (bin >= threshold)
	  {
	    inv = 1.F/bin;
	    (*all_segments_inv[segment_num])[view_num][axial_pos_num][tang_pos_num]  = inv;
	  }
	  else
	  {	  
	    inv =1/threshold;
	    (*all_segments_inv[segment_num])[view_num][axial_pos_num][tang_pos_num] = inv;
	  }
	  
	}
	
}

void 
find_inverse(ProjDataFromStream* proj_data_ptr_out, ProjDataFromStream* proj_data_ptr_in)
{  
  const ProjDataInfo * proj_data_info_ptr =
    proj_data_ptr_in->get_proj_data_info_ptr();
  
  const ProjDataInfo * proj_data_info_ptr_out =
    proj_data_ptr_out->get_proj_data_info_ptr();
  
  //const ProjDataFromStream*  projdatafromstream_in = 
   // dynamic_cast< const ProjDataFromStream*>(proj_data_ptr_in);
  
  //ProjDataFromStream*  projdatafromstream_out = 
  //  dynamic_cast<ProjDataFromStream*>(proj_data_ptr_out);
  float inv;
  float bin;

  //SegmentByView<float> segment_by_view = 
//	projdatafromstream_in->get_segment_by_view(0);

  //const float max_in_viewgram = segment_by_view.find_max();

  for (int segment_num = proj_data_info_ptr->get_min_segment_num(); 
  segment_num<= proj_data_info_ptr->get_max_segment_num();
  segment_num++)

 
    for ( int view_num = proj_data_info_ptr->get_min_view_num();
    view_num<=proj_data_info_ptr->get_max_view_num(); view_num++)
    {
      Viewgram<float> viewgram_in = proj_data_ptr_in->get_viewgram(view_num,segment_num);
      Viewgram<float> viewgram_out = proj_data_ptr_out->get_empty_viewgram(view_num,segment_num);

      // the following const was found in the ls_cyl.hs and the same value will
      // be used for thresholding both kappa_0 and kappa_1.
      // threshold  = 10^-4/max_in_sinogram
      // TODO - find out batter way of finding the threshold
      //const float max_in_viewgram = 0.01675F;
     //const float max_in_viewgram = 33.0262F;
      const float max_in_viewgram = 33.52F;

	//segment_by_view.find_max();
      //cerr << " Max number in viewgram is:" << max_in_viewgram;
	//= 33.0262F;
      const float threshold = 0.0001F*max_in_viewgram;    
   // const float threshold = 0.0000001F/max_in_viewgram;    
      

      //cerr << " Print out the threshold" << endl;
      //cerr << threshold << endl;
      
      for (int i= viewgram_in.get_min_axial_pos_num(); i<=viewgram_in.get_max_axial_pos_num();i++)
	for (int j= viewgram_in.get_min_tangential_pos_num(); j<=viewgram_in.get_max_tangential_pos_num();j++)
	{
	  bin= viewgram_in[i][j];
	 	 	 
	  if (bin >= threshold)
	  {
	    inv = 1.F/bin;
	    viewgram_out[i][j] = inv;
	  }
	 else
	  {	  
	   inv =1/threshold;
	   viewgram_out[i][j] = inv;
	  }

    	}
	proj_data_ptr_out->set_viewgram(viewgram_out);
    }

   //const string output_file_name_inside_inverse = "inside_inverse.s";
  // write_basic_interfile_PDFS_header(output_file_name_inside_inverse,*proj_data_ptr_out);
 //  cerr << "Finished" << endl;
    
}


void
do_segments_bck(DiscretisedDensity<3,float>& image, 
	    //  shared_ptr<ProjDataFromStream> proj_data_org,
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_view, const int end_view,
	    const int start_tang_pos_num,const int end_tang_pos_num,	   
	    BackProjectorByBin* back_projector_ptr,
	    bool fill_with_1)
{
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector_ptr->get_symmetries_used()->clone();  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)
  { 
    ViewSegmentNumbers vs(view, segment_num);
    symmetries_sptr->find_basic_view_segment_numbers(vs);
    if (find(already_processed.begin(), already_processed.end(), vs)
        != already_processed.end())
      continue;

    already_processed.push_back(vs);
    
    cerr << "Processing view " << vs.view_num()
      << " of segment " <<vs.segment_num()
      << endl;
    
    if(fill_with_1 )
    {
      RelatedViewgrams<float> viewgrams_empty= 
	proj_data_org.get_empty_related_viewgrams(vs, symmetries_sptr);
	//proj_data_org.get_empty_related_viewgrams(vs.view_num(),vs.segment_num(), symmetries_sptr);
      
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams_empty.begin();
      while(r_viewgrams_iter!=viewgrams_empty.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	if (start_view <= single_viewgram.get_view_num() && 
	  single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	{
	  single_viewgram.fill(1.F);
	}
	r_viewgrams_iter++;	  
      } 
      
      back_projector_ptr->back_project(image,viewgrams_empty,
	max(start_axial_pos_num, viewgrams_empty.get_min_axial_pos_num()), 
	min(end_axial_pos_num, viewgrams_empty.get_max_axial_pos_num()),
	start_tang_pos_num, end_tang_pos_num);
    }
    else
    {
      RelatedViewgrams<float> viewgrams = 
	proj_data_org.get_related_viewgrams(vs,
	//proj_data_org.get_related_viewgrams(vs.view_num(),vs.segment_num(),
	symmetries_sptr);
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
      
      while(r_viewgrams_iter!=viewgrams.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	{	  
	  if (start_view <= single_viewgram.get_view_num() && 
	    single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	  {
	    // ok
	  }
	  else
	  { 
	    // set to 0 to prevent it being backprojected
	    single_viewgram.fill(0);
	  }
	}
	++r_viewgrams_iter;
      }
	
      back_projector_ptr->back_project(image,viewgrams,
	  max(start_axial_pos_num, viewgrams.get_min_axial_pos_num()), 
	  min(end_axial_pos_num, viewgrams.get_max_axial_pos_num()),
	  start_tang_pos_num, end_tang_pos_num);      
    } // fill
  } // for view_num, segment_num    
    
}


#endif


template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& density) const
{
  DiscretisedDensity<3,elemT>* tmp_density =
      density.clone();
  virtual_apply(density, *tmp_density);
  delete tmp_density;
}


template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>::set_defaults()
{
  filter_coefficients.fill(0);
  proj_data_filename ="1";
  proj_data_ptr = NULL;
  attenuation_proj_data_filename ="1";
  attenuation_proj_data_ptr = NULL;
  mask_size = 0;
 
}

template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: initialise_keymap()
{
  parser.add_start_key("Modified Inverse Image Filter Parameters");
  parser.add_key("filter_coefficients", &filter_coefficients_for_parsing);
  parser.add_key("proj_data_filename", &proj_data_filename);
  parser.add_key("attenuation_proj_data_filename", &attenuation_proj_data_filename);
  parser.add_key("mask_size", &mask_size);
  parser.add_stop_key("END Modified Inverse Image Filter Parameters");
}

template <typename elemT>
bool 
ModifiedInverseAverigingImageFilter<elemT>::
post_processing()
{
  const unsigned int size = filter_coefficients_for_parsing.size();
  const int min_index = -(size/2);
  filter_coefficients.grow(min_index, min_index + size - 1);
  for (int i = min_index; i<= filter_coefficients.get_max_index(); ++i)
    filter_coefficients[i] = 
      static_cast<float>(filter_coefficients_for_parsing[i-min_index]);
  return false;
}


const char * const 
ModifiedInverseAverigingImageFilter<float>::registered_name =
  "Modified Inverse Image Filter";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need t

template ModifiedInverseAverigingImageFilter<float>;



END_NAMESPACE_TOMO

#endif
