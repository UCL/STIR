//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of non-static methods of ForwardProjectorByBinUsingRayTracing.

  \author Kris Thielemans
  \author Claire Labbe
  \author Damiano Belluzzo
  \author (based originally on C code by Matthias Egger)
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "Viewgram.h"
#include "RelatedViewgrams.h"
#include "VoxelsOnCartesianGrid.h"
#include "IndexRange4D.h"

START_NAMESPACE_TOMO

ForwardProjectorByBinUsingRayTracing::
  ForwardProjectorByBinUsingRayTracing(
				   const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                                   const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
 : symmetries(proj_data_info_ptr, image_info_ptr)
{}


const DataSymmetriesForViewSegmentNumbers * 
ForwardProjectorByBinUsingRayTracing::get_symmetries_used() const
{
  return &symmetries; 
}

void 
ForwardProjectorByBinUsingRayTracing::
actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& density,
		     const int min_axial_pos_num, const int max_axial_pos_num,
		     const int min_tangential_pos_num, const int max_tangential_pos_num)

{
  // this will throw an exception when the cast does not work
  const VoxelsOnCartesianGrid<float>& image = 
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);
  // TODO somehow check symmetry object in RelatedViewgrams

  const int num_views = viewgrams.get_proj_data_info_ptr()->get_num_views();
  RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
  if (viewgrams.get_basic_segment_num() == 0)
  {
    
    Viewgram<float> & pos_view = *r_viewgrams_iter;
    r_viewgrams_iter++;
    Viewgram<float> & pos_plus90 =*r_viewgrams_iter;

    //Viewgram<float> & pos_view = viewgrams.get_viewgram_reference(0); 
    //Viewgram<float> & pos_plus90 = viewgrams.get_viewgram_reference(1); 
    if (viewgrams.get_num_viewgrams() == 2)
    {
      forward_project_view_plus_90_and_delta_2D(
	pos_view, pos_plus90,  
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
    else
    {
      assert(viewgrams.get_basic_view_num() != 0);
      assert(viewgrams.get_basic_view_num() != num_views/4);
      r_viewgrams_iter=viewgrams.begin();
      r_viewgrams_iter+=2;
      Viewgram<float> & pos_min180 =*r_viewgrams_iter;

      //Viewgram<float> & pos_min180 = viewgrams.get_viewgram_reference(2); 
      //Viewgram<float> & pos_min90 = viewgrams.get_viewgram_reference(3); 
      r_viewgrams_iter++;
      Viewgram<float> & pos_min90 =*r_viewgrams_iter;
      forward_project_all_symmetries_2D(
	pos_view, pos_plus90, 
	pos_min180, pos_min90, 
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
  }
  else
  {
    // segment symmetry
    r_viewgrams_iter=viewgrams.begin();
    Viewgram<float> & pos_view = *r_viewgrams_iter; //0
    r_viewgrams_iter++;
    Viewgram<float> & neg_view =*r_viewgrams_iter; //1
    r_viewgrams_iter++;
    Viewgram<float> & pos_plus90 =*r_viewgrams_iter; //2
    r_viewgrams_iter++;
    Viewgram<float> & neg_plus90 =*r_viewgrams_iter ; //3
    if (viewgrams.get_num_viewgrams() == 4)
    {
      forward_project_view_plus_90_and_delta(
	pos_view, neg_view, pos_plus90, neg_plus90, 
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
    else
    {
      assert(viewgrams.get_basic_view_num() != 0);
      assert(viewgrams.get_basic_view_num() != num_views/4);
      //r_viewgrams_iter=viewgrams.begin();
      r_viewgrams_iter++;//4
      Viewgram<float> & pos_min180 =*r_viewgrams_iter;
      r_viewgrams_iter++;//5
      Viewgram<float> & neg_min180=*r_viewgrams_iter;
      r_viewgrams_iter++;//6
      Viewgram<float> & pos_min90=*r_viewgrams_iter;
	r_viewgrams_iter++;//7
      Viewgram<float> & neg_min90 =*r_viewgrams_iter;

      //Viewgram<float> & pos_min180 = viewgrams.get_viewgram_reference(4); 
      //Viewgram<float> & neg_min180 = viewgrams.get_viewgram_reference(5); 
      //Viewgram<float> & pos_min90 = viewgrams.get_viewgram_reference(6); 
      //Viewgram<float> & neg_min90 = viewgrams.get_viewgram_reference(7);     

      forward_project_all_symmetries(
	pos_view, neg_view, pos_plus90, neg_plus90, 
	pos_min180, neg_min180, pos_min90, neg_min90,
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
  }


}

/*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
*/

void 
ForwardProjectorByBinUsingRayTracing::
forward_project_all_symmetries(
			       Viewgram<float> & pos_view, 
			       Viewgram<float> & neg_view, 
			       Viewgram<float> & pos_plus90, 
			       Viewgram<float> & neg_plus90, 
			       Viewgram<float> & pos_min180, 
			       Viewgram<float> & neg_min180, 
			       Viewgram<float> & pos_min90, 
			       Viewgram<float> & neg_min90, 
			       const VoxelsOnCartesianGrid<float>& image,
			       const int min_ring_num, const int max_ring_num,
			       const int min_tangential_pos_num, const int max_tangential_pos_num)
{

  const ProjDataInfoCylindricalArcCorr * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr *>
    (pos_view.get_proj_data_info_ptr());
  if (proj_data_info_ptr == NULL)
    error("ForwardProjectorByBinUsingRayTracing::forward_project called with wrong type of ProjDataInfo\n");
    
  const int nviews = pos_view.get_proj_data_info_ptr()->get_num_views(); 
  const int view90 = nviews / 2;
  const int view45 = view90 / 2;
  
  const int segment_num = pos_view.get_segment_num();
  const float delta = proj_data_info_ptr->get_average_ring_difference(segment_num);  
  const int view = pos_view.get_view_num();

  assert(delta > 0);
  // relax 2 assertions to not break the temporary 4 parameter forward_project below
  assert(view >= 0);
  assert(view <= 2*view45);
  
  // KT 21/05/98 added some assertions
  assert(pos_plus90.get_view_num() == nviews / 2 + pos_view.get_view_num());
  /* remove 2 assertions which would break the temporary 4 parameter forward_project below
  assert(pos_min90.get_view_num() == nviews / 2 - pos_view.get_view_num());
  assert(pos_min180.get_view_num() == nviews - pos_view.get_view_num());
  */
  assert(neg_view.get_view_num() == pos_view.get_view_num());
  assert(neg_plus90.get_view_num() == pos_plus90.get_view_num());
  assert(neg_min90.get_view_num() == pos_min90.get_view_num());
  assert(neg_min180.get_view_num() == pos_min180.get_view_num());
  
  assert( pos_view.get_num_tangential_poss() ==image.get_x_size());
  assert( image.get_min_x() == -image.get_max_x());
  assert(image.get_voxel_size().x() == proj_data_info_ptr->get_tangential_sampling());
  // TODO rewrite code a bit to get rid of next restriction
  assert(image.get_min_z() == 0);
  
  assert(pos_view.get_max_tangential_pos_num() == -pos_view.get_min_tangential_pos_num());
  assert(pos_view.get_max_tangential_pos_num() == neg_view.get_max_tangential_pos_num());
  assert(pos_view.get_min_tangential_pos_num() == neg_view.get_min_tangential_pos_num());
  assert(delta ==
    -proj_data_info_ptr->get_average_ring_difference(neg_view.get_segment_num()));
  
  // KT 21/05/98 added const where possible
  // TODO C value depends whether you are in Double or not,
  // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
  const int C=1;
  
  int  D, s;
  int ax_pos0, my_ax_pos0;
  const float R = proj_data_info_ptr->get_ring_radius();
  
  // TODO replace
  const float itophi = _PI / nviews;
  
  const float cphi = cos(view * itophi);
  const float sphi = sin(view * itophi);
  
  const float 
    num_planes_per_virtual_ring = 
      proj_data_info_ptr->get_axial_sampling(segment_num)/image.get_voxel_size().z();  
  
  const 
    float num_virtual_rings_per_physical_ring = 
      proj_data_info_ptr->get_ring_spacing() /
      proj_data_info_ptr->get_axial_sampling(segment_num);
    
  // find correspondence between ring coordinates and image coordinates:
  // z = num_planes_per_virtual_ring * ring + virtual_ring_offset
  // compute the offset by matching up the centre of the scanner 
  // in the 2 coordinate systems
  const float num_planes_per_physical_ring = num_planes_per_virtual_ring*num_virtual_rings_per_physical_ring;
    
  const float virtual_ring_offset = 
    (image.get_max_z() + image.get_min_z())/2.F
    - num_planes_per_virtual_ring
    *(proj_data_info_ptr->get_max_axial_pos_num(segment_num) + num_virtual_rings_per_physical_ring*delta 
      + proj_data_info_ptr->get_min_axial_pos_num(segment_num))/2;
   
// CL 180298 Change the assignment as it is not exact due to symetries
// DB 24/4/98 changed to pos_view
    const int   projrad = (int) (pos_view.get_num_tangential_poss() / 2) - 1;  // CL 180298 SHould be smaller due to symetries test
 
    start_timers();

 
        // CL 200398 Move down to the case of other's phi some declarations
    Array <4,float> Projall(IndexRange4D(min_ring_num, max_ring_num, 0, 1, 0, 1, 0, 3));
        // KT 21/05/98 removed as now automatically zero 
        // Projall.fill(0);

    // TODO change comments
        // A loop which takes into account that axial ring size = 2*voxel z_size
        // TODO this will have to be changed when using CTI span!=1 data
        // This loop runs over 2 values offset = -0.25 and +0.25
        //CL 010399 Add two variables offset_start and offset_incr in order to change the
        //2 offset values in case of span!=1
    float offset_start = -.25F;
    float offset_incr = .5F;
    // CL&KT 21/12/99 new variable
    int num_lors_per_virtual_ring = 2;
    
    if (num_planes_per_virtual_ring == 1)
    {
        offset_start = 0;
        offset_incr=1;
	num_lors_per_virtual_ring = 1;
            //    cout << " Forwrad projection in case of span=1 " << endl;
    }
        //else cout << " Forwrad projection in case of span=1 " << endl;
    
    
    for (float offset = offset_start; offset < 0.3; offset += offset_incr)//SPAN
    {
        if (view == 0 || view == view45 ) {	/* phi=0 or 45 */
            for (D = 0; D < C; D++) {
                    /* Here s=0 and phi=0 or 45*/

                proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
                            delta + D, 0, R,min_ring_num, max_ring_num,
                            offset, 2, num_planes_per_virtual_ring, virtual_ring_offset );
                for (ax_pos0 = min_ring_num; ax_pos0 <= max_ring_num; ax_pos0++) {
                    my_ax_pos0 = C * ax_pos0 + D;
                        //CL 071099 Remove 0.5* and replace by num_lors_per_virtual_ring
                    pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
                    pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
                    neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
                    neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
                }
                    /* Now s!=0 and phi=0 or 45 */
                for (s = 1; s <= projrad; s++) {
                    proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
                                delta + D, s, R,min_ring_num, max_ring_num,
                                offset, 1, num_planes_per_virtual_ring, virtual_ring_offset);
                    for (ax_pos0 = min_ring_num; ax_pos0 <= max_ring_num; ax_pos0++) {
                        my_ax_pos0 = C * ax_pos0 + D;
                        pos_view[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
                        pos_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
                        pos_view[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][0]/ num_lors_per_virtual_ring; 
                        pos_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][2]/ num_lors_per_virtual_ring; 
                        neg_view[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
                        neg_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
                        neg_view[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][0]/ num_lors_per_virtual_ring; 
                        neg_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][2]/ num_lors_per_virtual_ring; 
                    }
                }
            }
        } else {

         
            for (D = 0; D < C; D++) {
                    /* Here s==0 and phi!=k*45 */
                proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi, 
                            delta + D, 0, R,min_ring_num, max_ring_num,
                            offset, 4, num_planes_per_virtual_ring, virtual_ring_offset );
                for (ax_pos0 = min_ring_num; ax_pos0 <= max_ring_num; ax_pos0++) {
                    my_ax_pos0 = C * ax_pos0 + D;
                    pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
                    pos_min90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][1]/ num_lors_per_virtual_ring; 
                    pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
                    pos_min180[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][3]/ num_lors_per_virtual_ring; 
                    neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
                    neg_min90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][1]/ num_lors_per_virtual_ring; 
                    neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
                    neg_min180[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][3]/ num_lors_per_virtual_ring; 
                }

                    /* Here s!=0 and phi!=k*45. */
                for (s = 1; s <= projrad; s++) {
                    proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
                                delta + D, s, R,min_ring_num, max_ring_num,
                                offset, 3, num_planes_per_virtual_ring, virtual_ring_offset );
                    for (ax_pos0 = min_ring_num; ax_pos0 <= max_ring_num; ax_pos0++) {
                        my_ax_pos0 = C * ax_pos0 + D;
                        pos_view[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
                        pos_min90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][1]/ num_lors_per_virtual_ring; 
                        pos_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
                        pos_min180[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][3]/ num_lors_per_virtual_ring; 
                        pos_view[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][0]/ num_lors_per_virtual_ring; 
                        pos_min90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][1]/ num_lors_per_virtual_ring; 
                        pos_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][2]/ num_lors_per_virtual_ring; 
                        pos_min180[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][3]/ num_lors_per_virtual_ring; 
                        neg_view[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
                        neg_min90[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][1]/ num_lors_per_virtual_ring; 
                        neg_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
                        neg_min180[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][3]/ num_lors_per_virtual_ring; 
                        neg_view[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][0]/ num_lors_per_virtual_ring; 
                        neg_min90[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][1]/ num_lors_per_virtual_ring; 
                        neg_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][2]/ num_lors_per_virtual_ring; 
                        neg_min180[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][3]/ num_lors_per_virtual_ring; 
                    }   
                }     
            }


        }// end of } else {
    }// end of test for offset loop
  

  
    stop_timers();
  
}


/*
This function projects 4 viewgrams related by symmetry.
It will be used for view=0 or 45 degrees 

  Here 0<=view < num_views/2 (= 90 degrees)
*/

void 
ForwardProjectorByBinUsingRayTracing::
  forward_project_view_plus_90_and_delta(Viewgram<float> & pos_view, 
				         Viewgram<float> & neg_view, 
				         Viewgram<float> & pos_plus90, 
				         Viewgram<float> & neg_plus90, 
				         const VoxelsOnCartesianGrid<float> & image,
				         const int min_axial_pos_num, const int max_axial_pos_num,
				         const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  //assert(pos_view.get_average_ring_difference() > 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;


    forward_project_all_symmetries(
                    pos_view, 
                    neg_view, 
                    pos_plus90, 
                    neg_plus90, 
                    dummy,
                    dummy,
                    dummy,
                    dummy,
                    image,
                    min_axial_pos_num, max_axial_pos_num,
                    min_tangential_pos_num, max_tangential_pos_num);

}

#if 0

void ForwardProjectorByBinUsingRayTracing::forward_project_2D( Segment<float> &sinos,const VoxelsOnCartesianGrid<float>&image)
{
    forward_project_2D(sinos,image,
		      sinos.get_min_axial_pos(), sinos.get_max_axial_pos());
}


void ForwardProjectorByBinUsingRayTracing::forward_project_2D( Segment<float> &sinos, const VoxelsOnCartesianGrid<float> &image,
		       const int view)
{
    forward_project_2D(sinos,image, view,
		      sinos.get_min_axial_pos(), sinos.get_max_axial_pos());
}

void ForwardProjectorByBinUsingRayTracing::forward_project_2D( Segment<float> &sinos, const VoxelsOnCartesianGrid<float> &image,
		       const int rmin, const int rmax)
{
    for (int view=0; view <= sinos.get_num_views()/4; view++)
        forward_project_2D(sinos,image, view, rmin, rmax);
  
}

void ForwardProjectorByBinUsingRayTracing::forward_project_2D(Segment<float> &sinos,const VoxelsOnCartesianGrid<float>& image,
			const int view, const int rmin, const int rmax)
{

  int segment_num =sinos.get_segment_num();

  const ProjDataInfoCylindricalArcCorr* proj_data_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(sinos.get_proj_data_info_ptr());

  if ( proj_data_cyl_ptr==NULL)
  {
    error("ForwardProjectorByBinUsingRayTracing::Casting failed\n");
  }

  assert(proj_data_cyl_ptr->get_average_ring_difference(segment_num) ==0);

  // CL&KT 05/11/98 use scan_info
  const float planes_per_virtual_ring = 
    sinos.get_proj_data_info_ptr()->get_scanner_ptr()->ring_spacing / image.get_voxel_size().z();
    
  int num_planes_per_virtual_ring = (int)(planes_per_virtual_ring + 0.5);
  assert(planes_per_virtual_ring > 0);
  // Check if planes_per_ring is very close to an int
  assert(fabs(num_planes_per_virtual_ring / planes_per_virtual_ring - 1) < 1e-5);

   // check if there is axial compression  
  if (proj_data_cyl_ptr->get_max_ring_difference(segment_num) != 0)
    {
      //TODO handle axial compression in a different way
      num_planes_per_virtual_ring = 1;
    }

  // We support only 1 or 2 planes per ring now
  assert(num_planes_per_virtual_ring == 1 || num_planes_per_virtual_ring == 2);

  // Initialise a 2D sinogram here to avoid reallocating it every ring
  // We use get_sinogram to get correct sizes etc., but 
  // data will be overwritten
  Sinogram<float> sino = sinos.get_sinogram(sinos.get_min_axial_pos());

  // First do direct planes
  {
    for (int ax_pos = rmin; ax_pos <= rmax; ax_pos++)
      {	
	sino = sinos.get_sinogram(ax_pos);
	forward_project_2D(sino,image, num_planes_per_virtual_ring*ax_pos, view);
	sinos.set_sinogram(sino);
      }
  }

  // Now do indirect planes
  if (num_planes_per_virtual_ring == 2)
    {
 
      // TODO terribly inefficient as this is repeated for every view
      // adding in lots of zeroes
      for (int  ax_pos = rmin; ax_pos < rmax; ax_pos++)
	{	
	  sino.fill(0);
	  // forward project the indirect plane
	  forward_project_2D(sino,image, num_planes_per_virtual_ring*ax_pos+1, view);
	  
	  // add 'half' of the sinogram to the 2 neighbouring rings
	  sino /= 2;

	  Sinogram<float> sino_tmp = sinos.get_sinogram(ax_pos);
	  sino_tmp += sino;
	  sinos.set_sinogram(sino_tmp);

	  sino_tmp = sinos.get_sinogram(ax_pos+1);
	  sino_tmp += sino;
	  sinos.set_sinogram(sino_tmp);
	}
    }
}

void ForwardProjectorByBinUsingRayTracing::forward_project_2D(Sinogram<float> &sino, const VoxelsOnCartesianGrid<float>& image,  
							      const int plane_num)
{
  for (int view=0; view <= sino.get_num_views()/4; view++)
    forward_project_2D(sino,image, plane_num, view);
}

void ForwardProjectorByBinUsingRayTracing::forward_project_2D(Sinogram<float> &sino,const VoxelsOnCartesianGrid<float>&image, 
							      const int plane_num, const int view)
{
  const ProjDataInfoCylindricalArcCorr* proj_data_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(sino.get_proj_data_info_ptr());
  
  if ( proj_data_cyl_ptr==NULL)
  {
    error("ForwardProjectorByBinUsingRayTracing::Casting failed");
    
    int segmnet_num = sino.get_segment_num();
    
    // KT 06/10/98 added num_planes_per_virtual_ring stuff for normalisation later on
    assert(proj_data_cyl_ptr->get_average_ring_difference(segmnet_num) ==0);
    
    // KT&CL 21/12/99 changed name from planes_per_ring to planes_per_virtual_ring
    const float planes_per_virtual_ring = 
      sino.get_proj_data_info_ptr()->get_scanner_ptr()->ring_spacing / image.get_voxel_size().z();
    
    int num_planes_per_virtual_ring = (int)(planes_per_virtual_ring + 0.5);
    assert(planes_per_virtual_ring > 0);
    // Check if planes_per_ring is very close to an int
    assert(fabs(num_planes_per_virtual_ring / planes_per_virtual_ring - 1) < 1e-5);
    
    // check if there is axial compression  
    if (proj_data_cyl_ptr->get_max_ring_difference(segmnet_num) != 0)
    {
      //TODO find out about axial compression in a different way
      num_planes_per_virtual_ring = 1;
    }
    
    // We support only 1 or 2 planes per ring now
    assert(num_planes_per_virtual_ring == 1 || num_planes_per_virtual_ring == 2);
    
    // CL&KT 05/11/98 use scan_info
    const int nviews = sino.get_proj_data_info_ptr()->get_num_views();
    const int view90 = nviews / 2;
    const int view45 = view90 / 2;
    const int plus90 = view90+view;
    const int min180 = nviews-view;
    const int min90 = view90-view;
    
    
    assert(sino.get_num_tangential_poss()  == image.get_x_size());
    assert(image.get_min_x() == -image.get_max_x());
    // CL&KT 05/11/98 use scan_info, enable assert
    assert(image.get_voxel_size().x == sino.get_proj_data_info_ptr()->get_num_tangential_poss()); 
    assert(sino.get_max_tangential_pos_num() == -sino.get_min_tangential_pos_num());
    
    
    // TODO C value depends whether you are in Double or not,
    // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
    const int C=1;
    
    int  D, s;
    // CL&KT 05/11/98 use scan_info
    const float R = sino.get_proj_data_info_ptr()->get_scanner_ptr()->ring_radius;
    
    const float itophi = _PI / nviews;
    
    const float cphi = cos(view * itophi);
    const float sphi = sin(view * itophi);
    
    const int   projrad = (int) (sino.get_num_tangential_poss() / 2) - 1;
    
    start_timers();
    
    //TODO for the moment, just handle 1 plane and use some 3D variables 
    const int min_ax_pos = 0;
    const int max_ax_pos = 0;
    const float delta = 0;
    int ax_pos0;
    int my_ax_pos0;
    
    Array <4,float> Projall(min_ax_pos, max_ax_pos, 0, 1, 0, 1, 0, 3);
    // KT&CL 21/12/99 const int axial_compression=0;//CL 220299 Add axial_compression
    
    // only 1 value of offset for 2D case.
    // However, we have to divide projections by num_planes_per_virtual_ring
    // to get values equal to the (average) line integral
    
    // KT&CL 21/12/99 use num_planes_per_virtual_ring in the offset and in proj_Siddon
    // inside proj_Siddon z=num_planes_per_virtual_ring*offset, but that has 
    // to be plane_num, so we set offset accordingly
    const float offset = float(plane_num)/num_planes_per_virtual_ring;
    {
      
      if (view == 0 || view == view45 ) {	/* phi=0 or 45 */
	for (D = 0; D < C; D++) {
	  /* Here s=0 and phi=0 or 45*/
	  
	  // CL&KT 05/11/98 use scan_info
	  proj_Siddon( Projall,image,proj_data_cyl_ptr, cphi, sphi,
	    delta + D, 0, R,min_ax_pos, max_ax_pos, offset, 2, num_planes_per_virtual_ring, 0);//KT&CL 21/12/99 changed last 2 parameters
	  for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	    my_ax_pos0 = C * ax_pos0 + D;
	    sino[view][0] += Projall[ax_pos0][0][0][0] / num_planes_per_virtual_ring; 
	    sino[plus90][0] += Projall[ax_pos0][0][0][2] / num_planes_per_virtual_ring;
	  }
	  /* Now s!=0 and phi=0 or 45 */
	  for (s = 1; s <= projrad; s++) {
	    proj_Siddon(Projall,image,proj_data_cyl_ptr, cphi, sphi,
	      delta + D, s, R,min_ax_pos,max_ax_pos, offset, 1, num_planes_per_virtual_ring, 0);//KT&CL 21/12/99 changed last 2 parameters
	    for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	      ax_pos0 = C * ax_pos0 + D;
	      sino[view][s] += Projall[ax_pos0][0][0][0] / num_planes_per_virtual_ring;
	      sino[plus90][s] += Projall[ax_pos0][0][0][2] / num_planes_per_virtual_ring;
	      sino[view][-s] += Projall[ax_pos0][0][1][0] / num_planes_per_virtual_ring;
	      sino[plus90][-s] += Projall[ax_pos0][0][1][2] / num_planes_per_virtual_ring;
	    }
	  }
	}
      } else {
	
	
	for (D = 0; D < C; D++) {
	  /* Here s==0 and phi!=k*45 */
	  proj_Siddon(Projall,image,proj_data_cyl_ptr, cphi, sphi,
	    delta + D, 0, R,min_ax_pos,max_ax_pos, offset, 4, num_planes_per_virtual_ring, 0);//KT&CL 21/12/99 changed last 2 parameters
	  for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	    my_ax_pos0 = C * ax_pos0 + D;
	    sino[view][0] += Projall[ax_pos0][0][0][0] / num_planes_per_virtual_ring;
	    sino[min90][0] += Projall[ax_pos0][0][0][1] / num_planes_per_virtual_ring;
	    sino[plus90][0] += Projall[ax_pos0][0][0][2] / num_planes_per_virtual_ring;
	    sino[min180][0] += Projall[ax_pos0][0][0][3] / num_planes_per_virtual_ring;
	  }
	  
	  /* Here s!=0 and phi!=k*45. */
	  for (s = 1; s <= projrad; s++) {
	    proj_Siddon(Projall,image,proj_data_cyl_ptr, cphi, sphi,
	      delta + D, s, R,min_ax_pos, max_ax_pos, offset, 3, num_planes_per_virtual_ring, 0);//KT&CL 21/12/99 changed last 2 parameters
	    for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	      my_ax_pos0 = C * ax_pos0 + D;
	      sino[view][s] += Projall[ax_pos0][0][0][0] / num_planes_per_virtual_ring;
	      sino[min90][s] += Projall[ax_pos0][0][0][1] / num_planes_per_virtual_ring;
	      sino[plus90][s] += Projall[ax_pos0][0][0][2] / num_planes_per_virtual_ring;
	      sino[min180][s] += Projall[ax_pos0][0][0][3] / num_planes_per_virtual_ring;
	      sino[view][-s] += Projall[ax_pos0][0][1][0] / num_planes_per_virtual_ring;
	      sino[min90][-s] += Projall[ax_pos0][0][1][1] / num_planes_per_virtual_ring;
	      sino[plus90][-s] += Projall[ax_pos0][0][1][2] / num_planes_per_virtual_ring;
	      sino[min180][-s] += Projall[ax_pos0][0][1][3] / num_planes_per_virtual_ring;
	    }   
	  }     
	}
	
	
      }// end of } else {
    }// end of test for offset loop
    
    
    
    stop_timers();
    
}





#endif


void 
ForwardProjectorByBinUsingRayTracing::
forward_project_view_plus_90_and_delta_2D(Viewgram<float> & pos_view, 
				          Viewgram<float> & pos_plus90, 
				         const VoxelsOnCartesianGrid<float> & image,
				         const int min_axial_pos_num, const int max_axial_pos_num,
				         const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  //assert(pos_view.get_average_ring_difference() > 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;


    forward_project_all_symmetries_2D(
                    pos_view, 
                    pos_plus90, 
                    dummy,
                    dummy,             
                    image,
                    min_axial_pos_num, max_axial_pos_num,
                    min_tangential_pos_num, max_tangential_pos_num);

}



void 
ForwardProjectorByBinUsingRayTracing::
forward_project_all_symmetries_2D(Viewgram<float> & pos_view, 
			         Viewgram<float> & pos_plus90, 
			         Viewgram<float> & pos_min180, 
			         Viewgram<float> & pos_min90, 
			         const VoxelsOnCartesianGrid<float>& image,
			         const int min_axial_pos_num, const int max_axial_pos_num,
			         const int min_tangential_pos_num, const int max_tangential_pos_num)
{

  const ProjDataInfoCylindricalArcCorr * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr *>
    (pos_view.get_proj_data_info_ptr());
  if (proj_data_info_ptr == NULL)
    error("ForwardProjectorByBinUsingRayTracing::forward_project called with wrong type of ProjDataInfo\n");
    
  const int nviews = pos_view.get_proj_data_info_ptr()->get_num_views(); 
  const int view90 = nviews / 2;
  const int view45 = view90 / 2;
  
  const int segment_num = pos_view.get_segment_num();
  const float delta = proj_data_info_ptr->get_average_ring_difference(segment_num);  
  const int view = pos_view.get_view_num();

  assert(delta == 0);
  // relax 2 assertions to not break the temporary 4 parameter forward_project below
  assert(view >= 0);
  assert(view <= 2*view45);
  
  // KT 21/05/98 added some assertions
  assert(pos_plus90.get_view_num() == nviews / 2 + pos_view.get_view_num());
  /* remove 2 assertions which would break the temporary 4 parameter forward_project below
  assert(pos_min90.get_view_num() == nviews / 2 - pos_view.get_view_num());
  assert(pos_min180.get_view_num() == nviews - pos_view.get_view_num());
  */
  //assert(neg_view.get_view_num() == pos_view.get_view_num());
  //assert(neg_plus90.get_view_num() == pos_plus90.get_view_num());
  //assert(neg_min90.get_view_num() == pos_min90.get_view_num());
  //assert(neg_min180.get_view_num() == pos_min180.get_view_num());
  
  assert( pos_view.get_num_tangential_poss() ==image.get_x_size());
  assert( image.get_min_x() == -image.get_max_x());
  assert(image.get_voxel_size().x() == proj_data_info_ptr->get_tangential_sampling());
  // TODO rewrite code a bit to get rid of next restriction
  assert(image.get_min_z() == 0);
  
  assert(pos_view.get_max_tangential_pos_num() == -pos_view.get_min_tangential_pos_num());
  //assert(pos_view.get_max_tangential_pos_num() == neg_view.get_max_tangential_pos_num());
 // assert(pos_view.get_min_tangential_pos_num() == neg_view.get_min_tangential_pos_num());
 // assert(delta ==
 //   -proj_data_info_ptr->get_average_ring_difference(neg_view.get_segment_num()));
  
  // KT 21/05/98 added const where possible
  // TODO C value depends whether you are in Double or not,
  // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
  const int C=1;
  
  int  D, s;
  int my_ax_pos0;
  const float R = proj_data_info_ptr->get_ring_radius();
  
  // TODO replace
  const float itophi = _PI / nviews;
  
  const float cphi = cos(view * itophi);
  const float sphi = sin(view * itophi);

  // TODO replace all this by stuff from DataSymmetries  
  const float 
    num_planes_per_virtual_ring = 
      proj_data_info_ptr->get_axial_sampling(segment_num)/image.get_voxel_size().z();  
  
  const 
    float num_virtual_rings_per_physical_ring = 
      proj_data_info_ptr->get_ring_spacing() /
      proj_data_info_ptr->get_axial_sampling(segment_num);
    
  // find correspondence between ring coordinates and image coordinates:
  // z = num_planes_per_virtual_ring * ring + virtual_ring_offset
  // compute the offset by matching up the centre of the scanner 
  // in the 2 coordinate systems
  const float num_planes_per_physical_ring = num_planes_per_virtual_ring*num_virtual_rings_per_physical_ring;
    
  const float virtual_ring_offset = 
    (image.get_max_z() + image.get_min_z())/2.F
    - num_planes_per_virtual_ring
    *(proj_data_info_ptr->get_max_axial_pos_num(segment_num) + num_virtual_rings_per_physical_ring*delta 
      + proj_data_info_ptr->get_min_axial_pos_num(segment_num))/2;
  
  // CL 180298 Change the assignment as it is not exact due to symetries
  // DB 24/4/98 changed to pos_view
  const int   projrad = (int) (pos_view.get_num_tangential_poss() / 2) - 1;  // CL 180298 SHould be smaller due to symetries test
  
  start_timers();
  
  Array <4,float> Projall(IndexRange4D(min_axial_pos_num, max_axial_pos_num, 0, 1, 0, 1, 0, 3));
  Array <4,float> Projall2(IndexRange4D(min_axial_pos_num, max_axial_pos_num+1, 0, 1, 0, 1, 0, 3));
  
  // What to do when num_planes_per_virtual_ring==2 ?
  // In the 2D case, the approach followed in 3D is ill-defined, as we would be 
  // forward projecting right along the edges of the voxels.
  // Instead, we take for the contribution to an axial_pos_num, 
  // 1/2 left_voxel + centre_voxel + 1/2 right_voxel
  
  int num_lors_per_virtual_ring = 2;
  
  if (num_planes_per_virtual_ring == 1)
  {
    num_lors_per_virtual_ring = 1;
  }
  
  
  
  if (view == 0 || view == view45 ) 
  {	/* phi=0 or 45 */
    for (D = 0; D < C; D++)       
    { /* Here s=0 and phi=0 or 45*/     
      {        
        proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
          delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
          0.F /*==offset*/, 2, num_planes_per_virtual_ring, virtual_ring_offset );
        for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
        {
          my_ax_pos0 = C * ax_pos0 + D;
          //CL 071099 Remove 0.5* and replace by num_lors_per_virtual_ring
          pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
          pos_plus90[my_ax_pos0][0] +=Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
          //neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
          //neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
        }
      }
      
      if (num_planes_per_virtual_ring == 2)
      {	 	  
        proj_Siddon(Projall2, image, proj_data_info_ptr, cphi, sphi,
          delta + D, 0, R, min_axial_pos_num,  max_axial_pos_num+1,
          -0.5F /*==offset*/, 2, num_planes_per_virtual_ring, virtual_ring_offset );
        for (int ax_pos0 =  min_axial_pos_num; ax_pos0 <=  max_axial_pos_num; ax_pos0++) 
        {
          my_ax_pos0 = C * ax_pos0 + D;
          //CL 071099 Remove 0.5* and replace by num_lors_per_virtual_ring
          pos_view[my_ax_pos0][0] += (Projall2[ax_pos0+1][0][0][0]+ Projall2[ax_pos0][0][0][0])/4; 
          pos_plus90[my_ax_pos0][0] += (Projall2[ax_pos0+1][0][0][2]+ Projall2[ax_pos0][0][0][2])/4; //CL 0710
          
        }
        
      }
      
      /* Now s!=0 and phi=0 or 45 */
      for (s = 1; s <= projrad; s++) 
      {
        {                              
          proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
            delta + D, s, R,min_axial_pos_num, max_axial_pos_num,
            0.F, 1, num_planes_per_virtual_ring, virtual_ring_offset);
          for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
          {
            my_ax_pos0 = C * ax_pos0 + D;
            pos_view[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
            pos_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
            pos_view[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][0]/ num_lors_per_virtual_ring; 
            pos_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][2]/ num_lors_per_virtual_ring; 
            //neg_view[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
            //neg_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
            //neg_view[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][0]/ num_lors_per_virtual_ring; 
            //neg_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][1][1][2]/ num_lors_per_virtual_ring; 
          }
        }
        if (num_planes_per_virtual_ring == 2)
        {                            
          proj_Siddon(Projall2, image, proj_data_info_ptr, cphi, sphi,
            delta + D, s, R,min_axial_pos_num, max_axial_pos_num+1,
            -0.5F, 1, num_planes_per_virtual_ring, virtual_ring_offset);
          for (int ax_pos0 =min_axial_pos_num; ax_pos0 <=max_axial_pos_num; ax_pos0++) 
          {
            my_ax_pos0 = C * ax_pos0 + D;
            pos_view[my_ax_pos0][s] +=(Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0])/4; 
            pos_plus90[my_ax_pos0][s] += (Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2])/4; 
            pos_view[my_ax_pos0][-s] +=(Projall2[ax_pos0][0][1][0]+Projall2[ax_pos0+1][0][1][0])/4; 
            pos_plus90[my_ax_pos0][-s] +=(Projall2[ax_pos0][0][1][2]+Projall2[ax_pos0+1][0][1][2])/4;
            
          }
        }
      } // Loop over s      
    } // Loop over D
  }
  else 
  {
    // general phi    
    for (D = 0; D < C; D++) 
    {
      /* Here s==0 and phi!=k*45 */
      {
        proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi, 
          delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
          0.F, 4, num_planes_per_virtual_ring, virtual_ring_offset );
        for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
        {
          my_ax_pos0 = C * ax_pos0 + D;
          pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
          pos_min90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][1]/ num_lors_per_virtual_ring; 
          pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
          pos_min180[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][3]/ num_lors_per_virtual_ring; 
          // neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
          // neg_min90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][1]/ num_lors_per_virtual_ring; 
          // neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]/ num_lors_per_virtual_ring; 
          // neg_min180[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][3]/ num_lors_per_virtual_ring; 
        }
      }
      
      if (num_planes_per_virtual_ring == 2)        
      {         
        proj_Siddon(Projall2, image, proj_data_info_ptr, cphi, sphi, 
          delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
          -0.5F, 4, num_planes_per_virtual_ring, virtual_ring_offset );
        for (int ax_pos0 = min_axial_pos_num; ax_pos0 <=max_axial_pos_num; ax_pos0++) 
        {
          my_ax_pos0 = C * ax_pos0 + D;
          pos_view[my_ax_pos0][0] +=  (Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0])/4; 
          pos_min90[my_ax_pos0][0] += (Projall2[ax_pos0][0][0][1]+Projall2[ax_pos0+1][0][0][1])/4; 
          pos_plus90[my_ax_pos0][0] +=(Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2])/4; 
          pos_min180[my_ax_pos0][0] +=(Projall2[ax_pos0][0][0][3]+Projall2[ax_pos0+1][0][0][3])/4; 
          // neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring; 
        }
      }
      
      /* Here s!=0 and phi!=k*45. */
      for (s = 1; s <= projrad; s++)         
      {
        {          
          proj_Siddon(Projall, image, proj_data_info_ptr, cphi, sphi,
            delta + D, s, R,min_axial_pos_num, max_axial_pos_num,
            0.F, 3, num_planes_per_virtual_ring, virtual_ring_offset );
          for (int ax_pos0 = min_axial_pos_num; ax_pos0<= max_axial_pos_num; ax_pos0++) 
          {
            my_ax_pos0 = C * ax_pos0 + D;
            pos_view[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][0]/ num_lors_per_virtual_ring; 
            pos_min90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][1]/ num_lors_per_virtual_ring; 
            pos_plus90[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][2]/ num_lors_per_virtual_ring; 
            pos_min180[my_ax_pos0][s] +=  Projall[ax_pos0][0][0][3]/ num_lors_per_virtual_ring; 
            pos_view[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][0]/ num_lors_per_virtual_ring; 
            pos_min90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][1]/ num_lors_per_virtual_ring; 
            pos_plus90[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][2]/ num_lors_per_virtual_ring; 
            pos_min180[my_ax_pos0][-s] +=  Projall[ax_pos0][0][1][3]/ num_lors_per_virtual_ring; 
            //neg_view[my_ax_pos0][s] +=  Projall[ax_pos0][1][0][0]/ num_lors_per_virtual_ring;             
          }   
        } 
        if (num_planes_per_virtual_ring == 2)
        {
          
          proj_Siddon(Projall2, image, proj_data_info_ptr, cphi, sphi,
            delta + D, s, R,min_axial_pos_num, max_axial_pos_num+1,
            -0.5F, 3, num_planes_per_virtual_ring, virtual_ring_offset );
          for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
          {
            my_ax_pos0 = C * ax_pos0 + D;
            pos_view[ my_ax_pos0][s] +=(Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0])/4; 
            pos_min90[my_ax_pos0][s] += (Projall2[ax_pos0][0][0][1]+Projall2[ax_pos0+1][0][0][1])/4; 
            pos_plus90[ my_ax_pos0][s] +=(Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2])/4; 
            pos_min180[ my_ax_pos0][s] += (Projall2[ax_pos0][0][0][3]+Projall2[ax_pos0+1][0][0][3])/4; 
            pos_view[ my_ax_pos0][-s] +=  (Projall2[ax_pos0][0][1][0] +Projall2[ax_pos0+1][0][1][0])/4; 
            pos_min90[ my_ax_pos0][-s] +=(Projall2[ax_pos0][0][1][1]+Projall2[ax_pos0+1][0][1][1])/4; 
            pos_plus90[ my_ax_pos0][-s] += (Projall2[ax_pos0][0][1][2]+ Projall2[ax_pos0+1][0][1][2])/4; 
            pos_min180[ my_ax_pos0][-s] += ( Projall2[ax_pos0][0][1][3]+ Projall2[ax_pos0+1][0][1][3])/4; 
            
          }   
        }     
        
      }// end of loop over s      
      
    }// end loop over D
  }// end of else
  
  
  
  stop_timers();
  
}





END_NAMESPACE_TOMO
