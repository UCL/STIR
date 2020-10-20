//
//
/*!
  \file
  \ingroup projection

  \brief Declares class stir::BackProjectorByBin

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
template <typename DataT> class DataProcessor;


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
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
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
#ifdef STIR_PROJECTORS_AS_V3
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
#endif
 /*! \brief projects the viewgrams into the volume
   it adds to the data backprojected since start_accumulating_in_new_target() was last called. */
 virtual void back_project(const ProjData&, int subset_num = 0, int num_subsets = 1);

 /*! \brief projects the viewgrams into the volume
  it adds to the data backprojected since start_accumulating_in_new_target() was last called. */
void back_project(const RelatedViewgrams<float>&);

 /*! \brief projects the specified range of the viewgrams and axial positions into the volume
  it adds to the data backprojected since start_accumulating_in_new_target() was last called. */
void back_project(const RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num);

 /*! \brief projects the specified range of the viewgrams, axial positions and tangential positions into the volume
   it adds to the data backprojected since start_accumulating_in_new_target() was last called. */
void back_project(const RelatedViewgrams<float>&,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num);

 /*! \brief tell the back projector to start accumulating into a new target.
   This function has to be called before any back-projection is initiated.*/
 virtual void start_accumulating_in_new_target();

 /*! \brief Get output
  This will overwrite the array-content of the argument with the result of all backprojections since calling `start_accumulating_in_new_target()`. Note that the argument has to have the same characteristics as what was used when calling `set_up()`.
 */
 virtual void get_output(DiscretisedDensity<3,float> &) const;

  /// Set data processor to use after back projection
  void set_post_data_processor(shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > post_data_processor_sptr);

protected:

  /*! \brief This actually does the back projection.
   There are two versions of this code to enable backwards compatibility.

   This is the older version (in which the backprojected image is not a member variable).
   In most cases, the new version (in which the backprojected image is a member variable) calls the old version.

   If you are developing your own projector, one of these two needs to be overloaded. It doesn't matter which,
   but it might as well be the new one in case we one day decide to remove the old ones.
  */
  virtual void actual_back_project(DiscretisedDensity<3,float>&,
                                   const RelatedViewgrams<float>&,
		                   const int min_axial_pos_num, const int max_axial_pos_num,
                           const int min_tangential_pos_num, const int max_tangential_pos_num);

  /*! \brief This actually does the back projection.
   There are two versions of this code to enable backwards compatibility.

   This is the newer version (in which the backprojected image is a member variable).
   In most cases, the new version calls the old version (in which the backprojected image is not a member variable).

   If you are developing your own projector, one of these two needs to be overloaded. It doesn't matter which,
   but it might as well be the new one in case we one day decide to remove the old ones.
  */
 virtual void actual_back_project(const RelatedViewgrams<float>&,
                          const int min_axial_pos_num, const int max_axial_pos_num,
                          const int min_tangential_pos_num, const int max_tangential_pos_num);
  //! check if the argument is the same as what was used for set_up()
  /*! calls error() if anything is wrong.

      If overriding this function in a derived class, you need to call this one.
   */
  virtual void check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3,float>& density_info) const;
  bool _already_set_up;

  //! Clone of the density sptr set with set_up()
  shared_ptr<DiscretisedDensity<3,float> > _density_sptr;
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > _post_data_processor_sptr;

  virtual void set_defaults();
  virtual void initialise_keymap();

 private:
  shared_ptr<const ProjDataInfo> _proj_data_info_sptr;

  void do_segments(DiscretisedDensity<3,float>& image, 
            const ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const int start_view, const int end_view);

#ifdef STIR_OPENMP
  //! A vector of back projected images that will be used with openMP. There will be as many images as openMP threads
  std::vector< shared_ptr<DiscretisedDensity<3,float> > > _local_output_image_sptrs;
#endif
};

END_NAMESPACE_STIR


#endif // __BackProjectorByBin_h_
