//
//
/*!
  \file
  \ingroup projection
  \ingroup NiftyPET

  \brief Back projection class using NiftyPET's GPU implementation.

  \author Richard Brown

  \todo NiftyPET limitations - currently limited
  to the Siemens mMR scanner and requires to CUDA.

  \todo STIR wrapper limitations - currently only
  projects all of the data (no subsets). NiftyPET
  currently supports spans 0, 1 and 11, but the STIR
  wrapper has only been tested for span-11.

  DOI - https://doi.org/10.1007/s12021-017-9352-y

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
#include "stir/recon_buildblock/NiftyPET_projector/NiftyPETHelper.h"

START_NAMESPACE_STIR

class DataSymmetriesForViewSegmentNumbers;

/*!
  \ingroup projection
  \brief Class for NiftyPET's GPU back projector

Only applicable for mMR data.
Current limitations:
 - Projects all of the data in one go
 - Only debugged for span 11.
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
  */
 virtual void set_up(		 
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
    );

  //! Symmetries not used, so returns TrivialDataSymmetriesForBins.
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

  /// Set use truncation - truncate before forward
  /// projection and after back projection
  void set_use_truncation(const bool use_truncation) { _use_truncation = use_truncation; }

protected:

 virtual void actual_back_project(const RelatedViewgrams<float>&,
                          const int min_axial_pos_num, const int max_axial_pos_num,
                          const int min_tangential_pos_num, const int max_tangential_pos_num);

 private:
  shared_ptr<DataSymmetriesForViewSegmentNumbers> _symmetries_sptr;
  NiftyPETHelper _helper;
  int _cuda_device;
  bool _cuda_verbosity;
  std::vector<float> _np_sino_w_gaps;
  bool _use_truncation;
};

END_NAMESPACE_STIR


#endif // __stir_gpu_BackProjectorByBinNiftyPET_h__
