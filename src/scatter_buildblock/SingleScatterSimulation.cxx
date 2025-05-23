/*
    Copyright (C) 2016, 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/scatter/SingleScatterSimulation.h"

START_NAMESPACE_STIR

const char * const
SingleScatterSimulation::registered_name =
        "PET Single Scatter Simulation";


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
    this->parser.add_start_key("PET Single Scatter Simulation Parameters");
    this->parser.add_stop_key("end PET Single Scatter Simulation Parameters");
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
set_up()
{
    // set to negative value such that this will be recomputed
    this->max_single_scatter_cos_angle = -1.F;

    return base_type::set_up();
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

