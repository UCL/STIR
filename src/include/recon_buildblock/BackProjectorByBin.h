//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declares class BackProjectorByBin

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$
*/
#ifndef __BackProjectorByBin_h_
#define __BackProjectorByBin_h_

#include "tomo/RegisteredObject.h"
#include "TimedObject.h"

START_NAMESPACE_TOMO

template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;
template <typename T> class shared_ptr;



/*!
  \ingroup recon_buildblock
  \brief Abstract base class for all back projectors
*/
class BackProjectorByBin : 
  public TimedObject,
  public RegisteredObject<BackProjectorByBin> 
{ 
public:

  //! Default constructor calls reset_timers()
 inline
   BackProjectorByBin();

  //! Stores all necessary geometric info
 /*! 
  If necessary, set_up() can be called more than once.

  Derived classes can assume that back_project()  will be called
  with input corresponding to the arguments of the last call to set_up(). 

  \warning there is currently no check on this.
  */
 virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    ) =0;

  /*! \brief Informs on which symmetries the projector handles
   It should get data related by at least those symmetries.
   Otherwise, a run-time error will occur (unless the derived
   class has other behaviour).
  */
 virtual  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const = 0;

  /*! \brief projects the viewgrams into the volume
   it adds to the data already present in the volume.*/
  inline
    void back_project(DiscretisedDensity<3,float>&,
                      const RelatedViewgrams<float>&);

  /*! \brief projects the specified range of the viewgrams into the volume
   it adds to the data already present in the volume.*/
  inline
    void back_project(DiscretisedDensity<3,float>&,
                      const RelatedViewgrams<float>&, 		  
		      const int min_axial_pos_num, const int max_axial_pos_num);

  /*! \brief projects the specified range of the viewgrams into the volume
   it adds to the data already present in the volume.*/
  inline
    void back_project(DiscretisedDensity<3,float>&,
                      const RelatedViewgrams<float>&,
		      const int min_axial_pos_num, const int max_axial_pos_num,
		      const int min_tangential_pos_num, const int max_tangential_pos_num);




  virtual ~BackProjectorByBin() {}



protected:

  virtual void actual_back_project(DiscretisedDensity<3,float>&,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
		                   const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;


};

END_NAMESPACE_TOMO

#include "BackProjectorByBin.inl"

#endif // __BackProjectorByBin_h_
