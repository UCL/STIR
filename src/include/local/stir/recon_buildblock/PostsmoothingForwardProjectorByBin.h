//
// $Id$
//
/*!

  \file

  \brief Declaration of class PostsmoothingForwardProjectorByBin

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#ifndef __Tomo_recon_buildblock_PostsmoothingForwardProjectorByBin__H__
#define __Tomo_recon_buildblock_PostsmoothingForwardProjectorByBin__H__

#include "tomo/RegisteredParsingObject.h"
#include "recon_buildblock/ForwardProjectorByBin.h"
#include "shared_ptr.h"
#include "VectorWithOffset.h"

START_NAMESPACE_TOMO

template <typename elemT> class Viewgram;

/*!
  \brief A very preliminary class that first forward projects, and then smooths the viewgrams

*/
class PostsmoothingForwardProjectorByBin : 
  public 
    RegisteredParsingObject<PostsmoothingForwardProjectorByBin,
                            ForwardProjectorByBin>
{
public:
  //! Name which will be used when parsing a PostsmoothingForwardProjectorByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  PostsmoothingForwardProjectorByBin();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );


  PostsmoothingForwardProjectorByBin(
                       const shared_ptr<ForwardProjectorByBin>& original_forward_projector_ptr,
                       const VectorWithOffset<float>& tangential_kernel,
		       const VectorWithOffset<float>& axial_kernel,
		       const bool smooth_segment_0_axially = false);

  // Informs on which symmetries the projector handles
  // It should get data related by at least those symmetries.
  // Otherwise, a run-time error will occur (unless the derived
  // class has other behaviour).
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;


private:

  shared_ptr<ForwardProjectorByBin> original_forward_projector_ptr;
  VectorWithOffset<float> tang_kernel;
  VectorWithOffset<float> ax_kernel;
  bool smooth_segment_0_axially;

  // next 2 necessary for parsing because of limitation in KeyParser
  vector<double> tang_kernel_double;
  vector<double> ax_kernel_double;

  void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num);

  void smooth(Viewgram<float>&,
              const int min_axial_pos_num, const int max_axial_pos_num,
              const int min_tangential_pos_num, const int max_tangential_pos_num) const;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_TOMO

#endif
