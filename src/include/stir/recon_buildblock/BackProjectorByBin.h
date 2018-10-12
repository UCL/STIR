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
    Copyright (C) 2018, University College London
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
  \warning Derived classes have to call set_up from the base class.
  */
 virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
    ) =0;

  /*! \brief Informs on which symmetries the projector handles
   It should get data related by at least those symmetries.
   Otherwise, a run-time error will occur (unless the derived
   class has other behaviour).
  */
 virtual  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const = 0;

 
  //! project whole proj_data into the volume
  /*! it overwrites the data already present in the volume.
  
      The optional arguments can be used to back-project only a subset of the data. 
      Subsets are determined as per detail::find_basic_vs_nums_in_subset(). However,
      this usage will likely be phased out at later stage.
    */
  void back_project(DiscretisedDensity<3,float>&,
		const ProjData&, int subset_num = 0, int num_subsets = 1);

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

 void back_project(DiscretisedDensity<3,float>&,
           const Bin&);

 virtual BackProjectorByBin* clone() const =0;


protected:

  virtual void actual_back_project(DiscretisedDensity<3,float>&,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
		                   const int min_tangential_pos_num, const int max_tangential_pos_num) = 0;

 virtual void actual_back_project(DiscretisedDensity<3,float>&,
                                  const Bin&) = 0;

 //! True if TOF has been activated.
 bool tof_enabled;

  //! check if the argument is the same as what was used for set_up()
  /*! calls error() if anything is wrong.

      If overriding this function in a derived class, you need to call this one.
   */
  virtual void check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3,float>& density_info) const;
  bool _already_set_up;

 private:
  shared_ptr<ProjDataInfo> _proj_data_info_sptr;
  //! The density ptr set with set_up()
  /*! \todo it is wasteful to have to store the whole image as this uses memory that we don't need. */
  shared_ptr<DiscretisedDensity<3,float> > _density_info_sptr;

  void do_segments(DiscretisedDensity<3,float>& image, 
            const ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const int start_view, const int end_view,
		const int start_timing_pos_num = 0, const int end_timing_pos_num = 0);


};

END_NAMESPACE_STIR


#endif // __BackProjectorByBin_h_
