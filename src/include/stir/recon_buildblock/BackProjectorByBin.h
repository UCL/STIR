//
//
/*!
  \file
  \ingroup projection

  \brief Declares class stir::BackProjectorByBin

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_BackProjectorByBin_h_
#define __stir_recon_buildblock_BackProjectorByBin_h_

#include "stir/RegisteredObject.h"
#include "stir/TimedObject.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;
class ProjData;
class DataSymmetriesForViewSegmentNumbers;



/*!
  \ingroup projection
  \brief Abstract base class for all back projectors
*/
class BackProjectorByBin : 
  public TimedObject,
  public RegisteredObject<BackProjectorByBin> 
{ 
public:

  //! Default constructor calls reset_timers()
  BackProjectorByBin();

  virtual ~BackProjectorByBin();

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

 
  //! project whole proj_data into the volume
  /*! it overwrites the data already present in the volume */
  void back_project(DiscretisedDensity<3,float>&,
	            const ProjData&);

  /*! \brief projects the viewgrams into the volume
   it adds to the data already present in the volume.*/
 void back_project(DiscretisedDensity<3,float>&,
		   const RelatedViewgrams<float>&);

  /*! \brief projects the specified range of the viewgrams into the volume
   it adds to the data already present in the volume.*/
 void back_project(DiscretisedDensity<3,float>&,
		   const RelatedViewgrams<float>&, 		  
		   const int min_axial_pos_num, const int max_axial_pos_num);

  /*! \brief projects the specified range of the viewgrams into the volume
    it adds to the data already present in the volume.*/
 void back_project(DiscretisedDensity<3,float>&,
		   const RelatedViewgrams<float>&,
		   const int min_axial_pos_num, const int max_axial_pos_num,
		   const int min_tangential_pos_num, const int max_tangential_pos_num);


protected:

  virtual void actual_back_project(DiscretisedDensity<3,float>&,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
		                   const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;
private:
  void do_segments(DiscretisedDensity<3,float>& image, 
            const ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const int start_view, const int end_view);


};

END_NAMESPACE_STIR


#endif // __BackProjectorByBin_h_
