/*
    Copyright (C) 2016 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::SingleScatterSimulation

  \author Nikos Efthimiou
*/

#ifndef __stir_scatter_SingleScatterSimulation_H__
#define __stir_scatter_SingleScatterSimulation_H__

#include "stir/Succeeded.h"
#include "stir/scatter/ScatterSimulation.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup scatter
  \brief PET single scatter simulation

  \todo The class is specific to PET so should be renamed accordingly.
*/
class SingleScatterSimulation : public RegisteredParsingObject<SingleScatterSimulation, ScatterSimulation, ScatterSimulation>
{
private:
  typedef RegisteredParsingObject<SingleScatterSimulation, ScatterSimulation, ScatterSimulation> base_type;

public:
  //! Name which will be used when parsing a ScatterSimulation object
  static const char* const registered_name;

  //! Default constructor
  SingleScatterSimulation();

  //! Constructor with initialisation from parameter file
  explicit SingleScatterSimulation(const std::string& parameter_filename);

  ~SingleScatterSimulation() override;

  Succeeded process_data() override;
  //! gives method information
  std::string method_info() const override;
  //! prompts the user to enter parameter values manually
  void ask_parameters() override;
  //! Perform checks and intialisations
  Succeeded set_up() override;

protected:
  void initialise(const std::string& parameter_filename);

  void set_defaults() override;
  void initialise_keymap() override;

  //! used to check acceptable parameter ranges, etc...
  bool post_processing() override;

  //!
  //! \brief simulate single scatter for one scatter point
  double simulate_for_one_scatter_point(const std::size_t scatter_point_num, const unsigned det_num_A, const unsigned det_num_B);

  double scatter_estimate(const Bin& bin) override;

  virtual void actual_scatter_estimate(double& scatter_ratio_singles, const unsigned det_num_A, const unsigned det_num_B);

private:
  //! larger angles will be ignored
  float max_single_scatter_cos_angle;
};

END_NAMESPACE_STIR

#endif
