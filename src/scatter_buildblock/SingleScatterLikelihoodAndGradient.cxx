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
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"

START_NAMESPACE_STIR

const char * const
SingleScatterLikelihoodAndGradient::registered_name =
        "Single Scatter Likelihood And Gradient";


SingleScatterLikelihoodAndGradient::
SingleScatterLikelihoodAndGradient() :
    base_type()
{
    this->set_defaults();
}

SingleScatterLikelihoodAndGradient::
SingleScatterLikelihoodAndGradient(const std::string& parameter_filename)
{
    this->initialise(parameter_filename);
}

SingleScatterLikelihoodAndGradient::
~SingleScatterLikelihoodAndGradient()
{}

void
SingleScatterLikelihoodAndGradient::
initialise_keymap()
{
    base_type::initialise_keymap();
    //    this->parser.add_start_key("Single Scatter Simulation");
    //    this->parser.add_stop_key("end Single Scatter Simulation");
}

void
SingleScatterLikelihoodAndGradient::
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
SingleScatterLikelihoodAndGradient::
set_defaults()
{
    base_type::set_defaults();
}

Succeeded
SingleScatterLikelihoodAndGradient::
set_up()
{
    return base_type::set_up();
}



void
SingleScatterLikelihoodAndGradient::
ask_parameters()
{
    base_type::ask_parameters();
}

bool
SingleScatterLikelihoodAndGradient::
post_processing()
{
    if (!base_type::post_processing())
        return false;
    return true;
}

END_NAMESPACE_STIR

