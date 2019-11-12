//
//
/*!
  \file
  \ingroup projection

  \brief Back projection class using NiftyPET's GPU implementation.

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
#ifndef __stir_gpu_BackProjectorByBinNiftyPET_h__
#define __stir_gpu_BackProjectorByBinNiftyPET_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/recon_buildblock/niftypet_projector/ProjectorByBinNiftyPETHelper.h"

START_NAMESPACE_STIR

class DataSymmetriesForViewSegmentNumbers;

/*!
  \ingroup projection
  \brief Class for NiftyPET's GPU back projector

Currently only projects all the data in one go.
*/
class BackProjectorByBinNiftyPET :
  public RegisteredParsingObject<BackProjectorByBinNiftyPET,
        BackProjectorByBin>
{ 
public:
    //! Name which will be used when parsing a BackProjectorByBin object
    static const char * const registered_name;

  //! Default constructor calls reset_timers()
  BackProjectorByBinNiftyPET();

  virtual ~BackProjectorByBinNiftyPET();

  /// Keymap
  virtual void initialise_keymap();

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
    );

  /*! \brief Informs on which symmetries the projector handles
   It should get data related by at least those symmetries.
   Otherwise, a run-time error will occur (unless the derived
   class has other behaviour).
  */
 virtual const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

  /// Back project
  void back_project(const ProjData&, int subset_num = 0, int num_subsets = 1);

 /// Get output
 virtual void get_output(DiscretisedDensity<3,float> &) const;


  /*! \brief tell the back projector to start accumulating into a new target.
    This function has to be called before any back-projection is initiated.*/
  virtual void start_accumulating_in_new_target();

  /// Set verbosity
  void set_verbosity(const bool verbosity) { _cuda_verbosity = verbosity; }

protected:

 virtual void actual_back_project(const RelatedViewgrams<float>&,
                          const int min_axial_pos_num, const int max_axial_pos_num,
                          const int min_tangential_pos_num, const int max_tangential_pos_num);

 private:
  shared_ptr<DataSymmetriesForViewSegmentNumbers> _symmetries_sptr;
  ProjectorByBinNiftyPETHelper _helper;
  int _cuda_device;
  bool _cuda_verbosity;
  std::vector<float> _np_sino_w_gaps;
};

END_NAMESPACE_STIR


#endif // __stir_gpu_BackProjectorByBinNiftyPET_h__
