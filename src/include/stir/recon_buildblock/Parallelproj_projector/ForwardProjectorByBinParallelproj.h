//
//

#ifndef __stir_gpu_ForwardProjectorByBinParallelproj_h__
#define __stir_gpu_ForwardProjectorByBinParallelproj_h__
/*!
  \file
  \ingroup Parallelproj

  \brief Forward projection class using Parallelproj's GPU implementation.

  \author Richard Brown
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"

START_NAMESPACE_STIR

class ProjDataInMemory;
class DataSymmetriesForViewSegmentNumbers;
namespace detail { class ParallelprojHelper; }

/*!
  \ingroup Parallelproj
  \brief Class for Parallelproj's forward projector.
*/
class ForwardProjectorByBinParallelproj:
  public RegisteredParsingObject<ForwardProjectorByBinParallelproj,
                                 ForwardProjectorByBin>
{ 
public:
  //! Name which will be used when parsing a ForwardProjectorByBin object
  static const char * const registered_name;

  //! Default constructor calls reset_timers()
  //inline
    ForwardProjectorByBinParallelproj();

    /// Constructor
    virtual ~ForwardProjectorByBinParallelproj();

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
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr // TODO should be Info only
    );

  //! Symmetries not used, so returns TrivialDataSymmetriesForBins.
  virtual  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

    /// Set input
    virtual void set_input(const DiscretisedDensity<3,float>&);

    /// Set verbosity
    void set_verbosity(const bool verbosity) { _cuda_verbosity = verbosity; }

    /// Set use truncation - truncate before forward
    /// projection and after back projection
    void set_use_truncation(const bool use_truncation) { _use_truncation = use_truncation; }

protected:

  virtual void actual_forward_project(RelatedViewgrams<float>& viewgrams,
          const int min_axial_pos_num, const int max_axial_pos_num,
          const int min_tangential_pos_num, const int max_tangential_pos_num) override;

private:
    shared_ptr<DataSymmetriesForViewSegmentNumbers> _symmetries_sptr;
    shared_ptr<ProjDataInMemory> _projected_data_sptr;
    shared_ptr<detail::ParallelprojHelper> _helper;
    bool _do_not_setup_helper;
    friend class ProjectorByBinPairUsingParallelproj;
    void set_helper(shared_ptr<detail::ParallelprojHelper>);
    int _cuda_device;
    bool _cuda_verbosity;
    bool _use_truncation;
};

END_NAMESPACE_STIR

#endif // __stir_gpu_ForwardProjectorByBinParallelproj_h__
