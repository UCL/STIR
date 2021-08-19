//
//
/*!
  \file
  \ingroup Parallelproj

  \brief Back projection class using Parallelproj's implementation.

  \author Richard Brown
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_gpu_BackProjectorByBinParallelproj_h__
#define __stir_gpu_BackProjectorByBinParallelproj_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
//#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"

START_NAMESPACE_STIR

class DataSymmetriesForViewSegmentNumbers;
class ProjDataInMemory;
namespace detail { class ParallelprojHelper; }

/*!
  \ingroup Parallelproj
  \brief Class for Parallelproj's back projector
*/
class BackProjectorByBinParallelproj :
  public RegisteredParsingObject<BackProjectorByBinParallelproj,
        BackProjectorByBin>
{ 
public:
    //! Name which will be used when parsing a BackProjectorByBin object
    static const char * const registered_name;

  //! Default constructor calls reset_timers()
  BackProjectorByBinParallelproj();

  virtual ~BackProjectorByBinParallelproj();

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

#if 0
 /// Back project
  void back_project(const ProjData&, int subset_num = 0, int num_subsets = 1);
#endif
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
  shared_ptr<ProjDataInMemory> _proj_data_to_backproject_sptr;
  shared_ptr<detail::ParallelprojHelper> _helper;
  bool _do_not_setup_helper;
  friend class ProjectorByBinPairUsingParallelproj;
  void set_helper(shared_ptr<detail::ParallelprojHelper>);
  int _cuda_device;
  bool _cuda_verbosity;
};

END_NAMESPACE_STIR


#endif // __stir_gpu_BackProjectorByBinParallelproj_h__
