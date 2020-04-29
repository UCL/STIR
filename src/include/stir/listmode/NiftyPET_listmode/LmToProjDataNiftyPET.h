#pragma once
//
//
/*!
  \file 
  \ingroup listmode

  \brief Wrapper to NiftyPET's listmode to projection data converter
 
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
    Copyright (C) 2020, University College of London
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

#include "stir/listmode/LmToProjDataAbstract.h"

START_NAMESPACE_STIR

/*!
  \ingroup listmode

  \brief This class is used to bin listmode data to projection data,
  i.e. (3d) sinograms using NiftyPET functionality.
*/

class ProjData;

class LmToProjDataNiftyPET : public LmToProjDataAbstract
{
public:

    /// Constructor
    LmToProjDataNiftyPET();

    /// Destructor
    virtual ~LmToProjDataNiftyPET() {}

    /// Set span
    void set_span(const int span)
    { _span = span; }

    /// Set CUDA device
    void set_cuda_device(const char cuda_device)
    { _cuda_device = cuda_device; }

    /// Set CUDA verbosity
    void set_cuda_verbosity(const bool cuda_verbosity)
    { _cuda_verbosity = cuda_verbosity; }

    /// Set start time
    void set_start_time(const int start_time)
    { _start_time = start_time; }

    /// Set stop time
    void set_stop_time(const int stop_time)
    { _stop_time = stop_time; }

    /// Set listmode binary file
    void set_listmode_binary_file(const std::string &listmode_binary_file)
    { _listmode_binary_file = listmode_binary_file; }

    /// Set norm binary file
    void set_norm_binary_file(const std::string &norm_binary_file)
    { _norm_binary_file = norm_binary_file; }

    /// This function does the actual work
    virtual void process_data();

    /// Get prompts
    shared_ptr<const ProjData> get_prompts_sptr() const
    { return _prompts_sptr; }

    /// Get delayeds
    shared_ptr<const ProjData> get_delayeds_sptr() const
    { return _delayeds_sptr; }

    /// Get randoms
    shared_ptr<const ProjData> get_randoms_sptr() const
    { return _randoms_sptr; }

    /// Get norm
    shared_ptr<const ProjData> get_norm_sptr() const
    {
        if (_norm_binary_file.empty())
            error("Set norm binary filename before extracting listmode.");
        return _norm_sptr;
    }

private:

    /// Check input values are as expected
    void check_input() const;

    int _span;
    char _cuda_device;
    bool _cuda_verbosity;
    int _start_time, _stop_time;
    std::string _listmode_binary_file;
    std::string _norm_binary_file;
    shared_ptr<ProjData> _prompts_sptr;
    shared_ptr<ProjData> _delayeds_sptr;
    shared_ptr<ProjData> _randoms_sptr;
    shared_ptr<ProjData> _norm_sptr;
};

END_NAMESPACE_STIR
