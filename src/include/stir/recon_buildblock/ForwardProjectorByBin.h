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
  \author Richard Brown

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018-2019, University College London
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
template <typename DataT> class DataProcessor;

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
#ifdef STIR_PROJECTORS_AS_V3
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
#endif
   //! project the volume into the whole proj_data
   /*! it overwrites the data already present in the projection data */
    virtual void forward_project(ProjData&,
                         int subset_num = 0, int num_subsets = 1, bool zero = true);

   //! project the volume into the viewgrams
   /*! it overwrites the data already present in the viewgram */
    void forward_project(RelatedViewgrams<float>&);

    void forward_project(RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num);

    void forward_project(RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

#if 0 // disabled as currently not used. needs to be written in the new style anyway
    //! function mainly used in ListMode reconstruction.
    /*! Calls actual_forward_project */
    void forward_project(Bin&,
                         const DiscretisedDensity<3,float>&);
#endif
    virtual ~ForwardProjectorByBin();

    /// Set input
    virtual void set_input(const DiscretisedDensity<3,float>&);

    /// Set data processor to use before forward projection. MUST BE CALLED BEFORE SET_INPUT.
    void set_pre_data_processor(shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > pre_data_processor_sptr);

protected:
  //! This virtual function has to be implemented by the derived class.
  virtual void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

  virtual void actual_forward_project(RelatedViewgrams<float>& viewgrams,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

#if 0 // disabled as currently not used. needs to be written in the new style anyway
    //! This virtual function has to be implemented by the derived class.
    virtual void actual_forward_project(Bin&,
                                        const DiscretisedDensity<3,float>&) = 0;
#endif

  //! check if the argument is the same as what was used for set_up()
  /*! calls error() if anything is wrong.

      If overriding this function in a derived class, you need to call this one.
   */
  virtual void check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3,float>& density_info) const;
  bool _already_set_up;

  //! The density ptr set with set_up()
  shared_ptr<DiscretisedDensity<3,float> > _density_sptr;
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > _pre_data_processor_sptr;

  virtual void set_defaults();
  virtual void initialise_keymap();

private:
  shared_ptr<ProjDataInfo> _proj_data_info_sptr;
};

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_ForwardProjectorByBin_h__
