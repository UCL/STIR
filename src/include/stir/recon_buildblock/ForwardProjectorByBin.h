//
// $Id$
//

#ifndef __stir_recon_buildblock_ForwardProjectorByBin_h__
#define __stir_recon_buildblock_ForwardProjectorByBin_h__
/*!
  \file
  \ingroup recon_buildblock

  \brief Base class for forward projectors which work on 'large' collections of bins:
  given the whole image, fill in a RelatedViewgrams<float> object.

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

   $Date$
   $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/RegisteredObject.h"
#include "stir/TimedObject.h"


START_NAMESPACE_STIR


template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;
template <typename T> class shared_ptr;


/*!
  \ingroup recon_buildblock
  \brief Abstract base class for all forward projectors
*/
class ForwardProjectorByBin : 
  public TimedObject,
  public RegisteredObject<ForwardProjectorByBin> 
{ 
public:

  //! Default constructor calls reset_timers()
  //inline
    ForwardProjectorByBin();

  //! Stores all necessary geometric info
 /*! 
  If necessary, set_up() can be called more than once.

  Derived classes can assume that forward_project()  will be called
  with input corresponding to the arguments of the last call to set_up(). 

  \warning there is currently no check on this.
  */
virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    ) =0;

  //! Informs on which symmetries the projector handles
  /*! It should get data related by at least those symmetries.
   Otherwise, a run-time error will occur (unless the derived
   class has other behaviour).
   */
  virtual  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const = 0;

  //! project the volume into the viewgrams
  /*! it overwrites the data already present in the viewgram */
    void forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&);

    void forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num);

    void forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num);

    virtual ~ForwardProjectorByBin();

protected:
  //! This virtual function has to be implemented by the derived class.
  virtual void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;

};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_ForwardProjectorByBin_h__
