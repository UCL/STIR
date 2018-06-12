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



END_NAMESPACE_STIR

