//
//

#ifndef __stir_gpu_ForwardProjectorByBinNiftyPET_h__
#define __stir_gpu_ForwardProjectorByBinNiftyPET_h__
/*!
  \file
  \ingroup projection

  \brief Class for forward projector with NiftyPET's GPU implementation.

  \author Richard Brown

*/
/*
    Copyright (C) 2019, University College London
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

#include "stir/recon_buildblock/ForwardProjectorByBin.h"

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
class ForwardProjectorByBinNiftyPET : 
  public TimedObject,
  public RegisteredObject<ForwardProjectorByBin> 
{ 
public:

  //! Default constructor calls reset_timers()
  //inline
    ForwardProjectorByBinNiftyPET();

  //! Stores all necessary geometric info
 /*! 
  If necessary, set_up() can be called more than once.

  Derived classes can assume that forward_project()  will be called
  with input corresponding to the arguments of the last call to set_up(). 

  \warning there is currently no check on this.
  \warning Derived classes have to call set_up from the base class.
  */
virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
    ) =0;

  //! Informs on which symmetries the projector handles
  /*! It should get data related by at least those symmetries.
   Otherwise, a run-time error will occur (unless the derived
   class has other behaviour).
   */
  virtual  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const = 0;

  //! project the volume into the whole or a subset of proj_data, optionally zeroing the rest
  /*! it overwrites the data already present in the projection data.
  
      The optional arguments can be used to project only a subset of the data. 
      Subsets are determined as per detail::find_basic_vs_nums_in_subset(). However,
      this usage will likely be phased out at later stage.*/
    void forward_project(ProjData&, 
			 const DiscretisedDensity<3,float>&, 
			 int subset_num = 0, int num_subsets = 1, bool zero = true);

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

   //! project the volume into the whole proj_data
   /*! it overwrites the data already present in the projection data */
    void forward_project(ProjData&);

   //! project the volume into the viewgrams
   /*! it overwrites the data already present in the viewgram */
    void forward_project(RelatedViewgrams<float>&);

    void forward_project(RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num);

    void forward_project(RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

    virtual ~ForwardProjectorByBinNiftyPET();

    /// Set input
    virtual void set_input(const shared_ptr<DiscretisedDensity<3,float> >&);

    /// Set input
    void set_input(const DiscretisedDensity<3,float>*);

protected:
  //! This virtual function has to be implemented by the derived class.
  virtual void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;

  virtual void actual_forward_project(RelatedViewgrams<float>& viewgrams,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

};

END_NAMESPACE_STIR

#endif // __stir_gpu_ForwardProjectorByBinNiftyPET_h__
