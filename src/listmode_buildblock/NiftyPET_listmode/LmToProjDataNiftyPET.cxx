/*!
  \file 
  \ingroup listmode

  \brief Implementation of class stir::LmToProjDataNiftyPET
 
  \author Richard Brown
*/
/*
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/NiftyPET_listmode/LmToProjDataNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/NiftyPETHelper.h"

START_NAMESPACE_STIR

LmToProjDataNiftyPET::LmToProjDataNiftyPET() : 
    _span(11), _cuda_device(0), _cuda_verbosity(true), _start_time(-1), _stop_time(-1), _norm_binary_file("")
{ }

void LmToProjDataNiftyPET::check_input() const
{
    if (_listmode_binary_file.empty())
        throw std::runtime_error("LmToProjDataNiftyPET::process_data: listmode binary file not set.");

    if (_start_time < 0)
        throw std::runtime_error("LmToProjDataNiftyPET::process_data: start time not set.");

    if (_stop_time < 0)
        throw std::runtime_error("LmToProjDataNiftyPET::process_data: stop time not set.");

    // Check span
    if (_span != 11)
        throw std::runtime_error("LmToProjDataNiftyPET::process_data: currently only implemented for span 11.");
}

void LmToProjDataNiftyPET::process_data()
{
    // Set up the niftyPET binary helper
    NiftyPETHelper helper;
    helper.set_cuda_device_id   ( _cuda_device  );
    helper.set_span             ( static_cast<char>(_span) );
    helper.set_att(0);
    helper.set_verbose(_cuda_verbosity);
    helper.set_scanner_type(Scanner::Siemens_mMR);
    helper.set_up();

    helper.lm_to_proj_data(_prompts_sptr, _delayeds_sptr,
                _randoms_sptr, _norm_sptr,
                _start_time, _stop_time,
                _listmode_binary_file , _norm_binary_file);
}

END_NAMESPACE_STIR
