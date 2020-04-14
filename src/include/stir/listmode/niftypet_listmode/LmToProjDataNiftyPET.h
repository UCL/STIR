#pragma once
//
//
/*!
  \file 
  \ingroup listmode

  \brief Wrapper to NiftyPET's listmode to projection data converter
 
  \author Richard Brown
  
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
    void set_cuda_verbosity(const int cuda_verbosity)
    { _cuda_verbosity = cuda_verbosity; }

    /// Set listmode binary file
    void set_listmode_binary_file(const std::string &listmode_binary_file)
    { _listmode_binary_file = listmode_binary_file; }

    /// Set start time
    void set_start_time(const int start_time)
    { _start_time = start_time; }

    /// Set stop time
    void set_stop_time(const int stop_time)
    { _stop_time = stop_time; }

    /// This function does the actual work
    virtual void process_data();

    /// Get output
    shared_ptr<ProjData> get_output() const
    { return output_sptr; }

private:

    /// Check input values are as expected
    void check_input() const;

    int _span;
    char _cuda_device;
    int _cuda_verbosity;
    std::string _listmode_binary_file;
    int _start_time, _stop_time;
    shared_ptr<ProjData> output_sptr;
};

END_NAMESPACE_STIR
