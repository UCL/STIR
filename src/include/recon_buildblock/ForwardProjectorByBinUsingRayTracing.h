//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock
  
  \brief Declaration of class ForwardProjectorByBinUsingRayTracing
    
  \author Kris Thielemans
  \author PARAPET project
      
   \date $Date$
   \version $Revision$
*/
#include "recon_buildblock/ForwardProjectorByBin.h"

START_NAMESPACE_TOMO

template <typename T> class shared_ptr;
template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <typename elemT> class VoxelsOnCartesianGrid;
template <int num_dimensions, typename elemT> class Array;
class ProjDataInfo;
class ProjDataInfoCylindricalArcCorr;
class DataSymmetriesForViewSegmentNumbers;

/*!
  \ingroup recon_buildblock
  \brief This class implements forward projection using Siddon's algorithm for
  ray tracing. That is, it computes length of intersection with the voxels.
*/
  /*TODOdoc

*/

class ForwardProjectorByBinUsingRayTracing : public ForwardProjectorByBin
{
public:


  ForwardProjectorByBinUsingRayTracing(
                       const shared_ptr<ProjDataInfo>&,
                       const shared_ptr<DiscretisedDensity<3,float> >&);

  // Informs on which symmetries the projector handles
  // It should get data related by at least those symmetries.
  // Otherwise, a run-time error will occur (unless the derived
  // class has other behaviour).
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;


private:
  void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num);


  const DataSymmetriesForViewSegmentNumbers * symmetries_ptr;
  /*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
    */

  virtual void 
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
				 const int min_axial_pos_num, const int max_axial_pos_num,
				 const int min_tangential_pos_num, const int max_tangential_pos_num);


  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees 
    (or all others if the above version is not implemented in 
    the derived class)
    Here 0<=view < num_views/2 (= 90 degrees)
    */

public:
  virtual void 
  forward_project_view_plus_90_and_delta(
					 Viewgram<float> & pos_view, 
					 Viewgram<float> & neg_view, 
					 Viewgram<float> & pos_plus90, 
					 Viewgram<float> & neg_plus90, 
					 const VoxelsOnCartesianGrid<float> & image,
					 const int min_axial_pos_num, const int max_axial_pos_num,
					 const int min_tangential_pos_num, const int max_tangential_pos_num); 
void forward_project_all_symmetries_2D(
			       Viewgram<float> & pos_view, 
			       Viewgram<float> & pos_plus90, 
			       Viewgram<float> & pos_min180, 
			       Viewgram<float> & pos_min90, 
			       const VoxelsOnCartesianGrid<float>& image,
			       const int min_axial_pos_num, const int max_axial_pos_num,
			       const int min_tangential_pos_num, const int max_tangential_pos_num);
void 
forward_project_view_plus_90_and_delta_2D(Viewgram<float> & pos_view, 
				          Viewgram<float> & pos_plus90, 
				          const VoxelsOnCartesianGrid<float> & image,
 				          const int min_axial_pos_num, const int max_axial_pos_num,
				          const int min_tangential_pos_num, const int max_tangential_pos_num);

//! The actual implementation of Siddon's algorithm 
  static void proj_Siddon(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
                 const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
                 const float cphi, const float sphi, const float delta, const int s, 
                 const float R, const int rmin, const int rmax, const float offset, 
                 const int Siddon,
                 const int num_planes_per_virtual_ring,
		 const float virtual_ring_offset);


};
END_NAMESPACE_TOMO
