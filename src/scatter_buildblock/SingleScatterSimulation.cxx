/*
    Copyright (C) 2016 University College London
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
#include "stir/scatter/SingleScatterSimulation.h"

START_NAMESPACE_STIR

const char * const
SingleScatterSimulation::registered_name =
        "Single Scatter Simulation";


SingleScatterSimulation::
SingleScatterSimulation() :
    base_type()
{
    this->set_defaults();
}

SingleScatterSimulation::
SingleScatterSimulation(const std::string& parameter_filename)
{
    this->initialise(parameter_filename);
}

SingleScatterSimulation::
~SingleScatterSimulation()
{}


void
SingleScatterSimulation::
initialise_keymap()
{
    base_type::initialise_keymap();
    //    this->parser.add_start_key("Single Scatter Simulation");
    //    this->parser.add_stop_key("end Single Scatter Simulation");
}

void
SingleScatterSimulation::
initialise(const std::string& parameter_filename)
{
    if (parameter_filename.size() == 0)
    {
        this->set_defaults();
        this->ask_parameters();
    }
    else
    {
        this->set_defaults();
        if (!this->parse(parameter_filename.c_str()))
        {
            error("Error parsing input file %s, exiting", parameter_filename.c_str());
        }
    }
}

void
SingleScatterSimulation::
set_defaults()
{
    base_type::set_defaults();
}

Succeeded
SingleScatterSimulation::
process_data()
{
    return base_type::process_data();
}

void
SingleScatterSimulation::
ask_parameters()
{
    base_type::ask_parameters();
}

bool
SingleScatterSimulation::
post_processing()
{
    if (!base_type::post_processing())
        return false;
    return true;
}

std::string
SingleScatterSimulation::
method_info() const
{
    return this->registered_name;
}


END_NAMESPACE_STIR

