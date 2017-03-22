//
//

#ifndef __stir_recon_buildblock_ForwardProjectorByBin_h__
#define __stir_recon_buildblock_ForwardProjectorByBin_h__
/*!
  \file
  \ingroup projection

  \brief Base class for forward projectors which work on 'large' collections of bins: given the whole image, fill in a stir::RelatedViewgrams<float> object.

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

#include "stir/RegisteredObject.h"
#include "stir/TimedObject.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/shared_ptr.h"
#include "stir/Bin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"

START_NAMESPACE_STIR


template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, class elemT> class DiscretisedDensity;
class ProjDataInfo;
class ProjData;
class DataSymmetriesForViewSegmentNumbers;


/*!
  \ingroup projection
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

  //! project the volume into the whole proj_data
  /*! it overwrites the data already present in the projection data */
    void forward_project(ProjData&, 
			 const DiscretisedDensity<3,float>& );

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

    //! Overloaded function mainly used in ListMode reconstruction.
    void forward_project(Bin&,
                         const DiscretisedDensity<3,float>&);

    //! This is a really ungly way to sort the extra bit of information which are needed for
    //! the TOF reconstruction. In this base class we are going to store a pointer the the current
    //! ProjMatrixElements row and the two detection points. Which the are going to be accessed by the
    //! child class in case that tof_enabled.
    //! The common ProjMatrixElems row will provide a 'bridge' between forward and back projection as,
    //! we'll be able to keep it and not calculate it again.
    void set_tof_data(const CartesianCoordinate3D<float>*,
                      const CartesianCoordinate3D<float>*);

    ProjMatrixElemsForOneBin* get_tof_row() const;

    virtual ~ForwardProjectorByBin();

protected:
  //! This virtual function has to be implemented by the derived class.
  virtual void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;

    //! This virtual function has to be implemented by the derived class.
    virtual void actual_forward_project(Bin&,
                                        const DiscretisedDensity<3,float>&) = 0;

    //! True if TOF has been activated.
    bool tof_enabled;

    shared_ptr<ProjMatrixElemsForOneBin> tof_probabilities;
    const CartesianCoordinate3D<float>* point1;
    const CartesianCoordinate3D<float>* point2;

};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_ForwardProjectorByBin_h__
