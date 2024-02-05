//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationPETFromComponents

  \author Kris Thielemans
*/
/*
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisationPETFromComponents_H__
#define __stir_recon_buildblock_BinNormalisationPETFromComponents_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/ML_norm.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
  \brief A BinNormalisation class that uses component-based normalisation for PET

  Components currently supported are crystal efficiencies, geometric factors
  (constrained by symmetry) and block data. The latter were introduced to
  cope with timing alignment issues between blocks, but are generally
  not recommended in the current estimation process (by ML_estimate_component_based_normalisation)
  as the model allows for too much freedom.

  The detection efficiency of a crystal pair is modelled as
  \f[
    \epsilon_i \epsilon_j g_{ij} B_{ij}
  \f]
  with \f$ i,j \f$ crystal indices, and \f$ g_{ij} \f$ obtained from a GeoData3D object
  by using symmetries, and \f$ B_{ij} \f$ from a BlockData3D object by finding which
  blocks the crystals belong to.

  Symmetries for the geometric factors are described in
  <br />
  Niknejad, T., Tavernier, S., Varela, J. and Thielemans, K.
  <i>Validation of 3D model-based maximum-likelihood estimation of normalisation factors for partial ring positron emission
  tomography</i>. 
  in 2016 IEEE Nuclear Science Symposium, Medical Imaging Conference and Room-Temperature Semiconductor Detector
  Workshop (NSS/MIC/RTSD) 1-5 (2016). doi:10.1109/NSSMIC.2016.8069577.
  <br /> 
  Note however that this describes rotational/translational symmetry per block, while the default is now to use
  symmetries per bucket ( see the \c do_symmetry_per_block argument of allocate()).
  (The block factors still work per block, not bucket).

  This class does not actually set the relevant factors. That is left to external
  methods via the crystal_efficiencies(), geometric_factors() and block_factors()
  members. If they are not set, the factor is not applied (i.e. assumed to be 1).

  The model is constructed for the "physical" crystals only. The "virtual"
  crystals are forced to have 0 detection efficiency.

  \todo This class should probably be derived from BinNormalisationWithCalibration.
  \todo The class currently does not handle "compressed" projection data (i.e. span etc).

  \todo Currently, set_up() creates a ProjDataInMemory object with the PET detection efficiencies.
  This uses a lot of memory unfortunately.

*/
class BinNormalisationPETFromComponents : public BinNormalisation
{
private:
  using base_type = BinNormalisation;

public:
  // need to have this here as it's a pure virtual in BinNormalisation.
  // it is normally implemented via RegisteredParsingObject, but
  // we currently do no derive from that.
  std::string get_registered_name() const override { return this->registered_name; }

  //! Default constructor
  /*!
    \warning You should not call any member functions for any object just
    constructed with this constructor. Initialise the object properly first.
  */
  BinNormalisationPETFromComponents();

  //! check if we would be multiplying with 1 (i.e. do nothing)
  /*! Checks if all data is equal to 1 (up to a tolerance of 1e-4). To do this, it checks if all components are 1.
   */
  bool is_trivial() const override;

  //! Checks if we can handle certain projection data.
  /*! Compares the stored ProjDataInfo  with the ProjDataInfo supplied. */
  Succeeded set_up(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>&) override;

  //! Normalise some data
  /*!
    This means \c divide with the efficiency model. 0/0 is set to 0.
  */

  void apply(RelatedViewgrams<float>& viewgrams) const override;

  using base_type::apply;
  //! Undo the normalisation of some data
  /*!
    This means \c multiply with the efficiency model.
  */
  void undo(RelatedViewgrams<float>& viewgrams) const override;
  using base_type::undo;
  float get_bin_efficiency(const Bin& bin) const override;

#if 0
  //! Get a shared_ptr to the normalisation proj_data.
  virtual shared_ptr<ProjData> get_norm_proj_data_sptr() const;
#endif

  //! Allocate the relevant factors
  /*! They are currently probably set to 0, but do not rely on this. */
  void allocate(shared_ptr<const ProjDataInfo>, bool do_eff, bool do_geo, bool do_block = false,
                bool do_symmetry_per_block = false);

  DetectorEfficiencies& crystal_efficiencies() { return efficiencies; }
  GeoData3D& geometric_factors() { return geo_data; }
  BlockData3D& block_factors() { return block_data; }

  //! Sets all factors to empty and flags that allocations need to be done
  /*! Also calls base_type::set_defaults() */
  void set_defaults() override;

protected:
  DetectorEfficiencies efficiencies;
  GeoData3D geo_data;
  BlockData3D block_data;

  //! \name checks if factors are initialised
  //@{
  bool has_crystal_efficiencies() const;
  bool has_geometric_factors() const;
  bool has_block_factors() const;
  //@}

private:
  static const char* const registered_name;
  shared_ptr<ProjDataInMemory> invnorm_proj_data_sptr;
  bool _already_allocated;
  bool _is_trivial;
  void create_proj_data();
};

END_NAMESPACE_STIR

#endif
