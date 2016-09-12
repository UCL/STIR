//
//
/*
    Copyright (C) 2016, UCL
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
/*!
  \file
  \ingroup scatter
  \brief inline functions of ScatterSimulation

  \author Nikos Efthimiou

*/
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/error.h"

START_NAMESPACE_STIR

/**************** Functions to set images ****************/

Succeeded
ScatterSimulation::
set_activity_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
    {
        warning("ScatterSimulation: Unable to set the activity image");
        return Succeeded::no;
    }

    this->activity_image_sptr = arg;
    this->remove_cache_for_integrals_over_activity();

    return Succeeded::yes;
}

void
ScatterSimulation::
set_activity_image(const std::string& filename)
{
    this->activity_image_filename = filename;
    this->activity_image_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);

    if (is_null_ptr(this->activity_image_sptr))
    {
        error(boost::format("Error reading activity image %s") %
              this->activity_image_filename);
    }
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_density_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image");
    this->density_image_sptr=arg;
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image(const std::string& filename)
{
    this->density_image_filename=filename;
    this->density_image_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);
    if (is_null_ptr(this->density_image_sptr))
    {
        error(boost::format("Error reading density image %s") %
              this->density_image_filename);
    }
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image_for_scatter_points_sptr(const shared_ptr<DiscretisedDensity<3,float> >& arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image for scatter points.");
    this->density_image_for_scatter_points_sptr = arg;
    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_density_image_for_scatter_points(const std::string& filename)
{
    this->density_image_for_scatter_points_filename=filename;
    this->density_image_for_scatter_points_sptr=
            read_from_file<DiscretisedDensity<3,float> >(filename);

    if (is_null_ptr(this->density_image_for_scatter_points_sptr))
    {
        error(boost::format("Error reading density_for_scatter_points image %s") %
              this->density_image_for_scatter_points_filename);
    }
    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
set_output_proj_data_sptr(const shared_ptr<ExamInfo>& _exam,
                          const shared_ptr<ProjDataInfo>& _info,
                          const std::string & filename)
{
    if (filename.size() > 0 )
        this->output_proj_data_sptr.reset(new ProjDataInterfile(_exam,
                                                                _info,
                                                                filename));
    else
        this->output_proj_data_sptr.reset( new ProjDataInMemory(_exam,
                                                                _info));
}

void
ScatterSimulation::
set_output_proj_data(const std::string& filename)
{
    this->output_proj_data_filename = filename;

    if (is_null_ptr(this->template_exam_info_sptr))
    {
        shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
        this->output_proj_data_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                                                this->proj_data_info_ptr->create_shared_clone(),
                                                                this->output_proj_data_filename));
    }
    else
        this->output_proj_data_sptr.reset(new ProjDataInterfile(this->template_exam_info_sptr,
                                                                this->proj_data_info_ptr->create_shared_clone(),
                                                                this->output_proj_data_filename));
}

void
ScatterSimulation::
get_output_proj_data(shared_ptr<ProjData>& arg)
{
    arg = this->output_proj_data_sptr;
}

void
ScatterSimulation::
set_template_proj_data_info_sptr(const shared_ptr<ProjDataInfo>& arg)
{
    this->proj_data_info_ptr = dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(arg.get());

    if (is_null_ptr(this->proj_data_info_ptr))
        error("ScatterEstimationByBin can only handle non-arccorrected data");

    this->proj_data_info_sptr = arg;

    // find final size of detection_points_vector
    this->total_detectors =
            this->proj_data_info_ptr->get_scanner_ptr()->get_num_rings()*
            this->proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring ();

    // reserve space to avoid reallocation, but the actual size will grow dynamically
    this->detection_points_vector.reserve(static_cast<std::size_t>(this->total_detectors));

    // remove any cached values as they'd be incorrect if the sizes changes
    this->remove_cache_for_integrals_over_attenuation();
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_template_proj_data_info(const std::string& filename)
{
    this->template_proj_data_filename = filename;
    shared_ptr<ProjData> template_proj_data_sptr =
            ProjData::read_from_file(this->template_proj_data_filename);

    this->set_template_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone());
    this->set_exam_info_sptr(template_proj_data_sptr->get_exam_info_ptr()->create_shared_clone());
}

void
ScatterSimulation::
set_exam_info_sptr(const shared_ptr<ExamInfo>& arg)
{
    this->template_exam_info_sptr = arg;
}

void
ScatterSimulation::
set_attenuation_threshold(const float& arg)
{
    attenuation_threshold = arg;
}

void
ScatterSimulation::
set_random_point(const bool& arg)
{
    random = arg;
}

void
ScatterSimulation::
set_cache_enabled(const bool &arg)
{
    use_cache = arg;
}


//
//
// NOT SETS THE REST
//
//


float
ScatterSimulation::
dif_Compton_cross_section(const float cos_theta, float energy)
{
    const double Re = 2.818E-13;   // aktina peristrofis electroniou gia to atomo tou H
    const double sin_theta_2= 1-cos_theta*cos_theta ;
    const double P= 1.0/(1.0+(energy/511.0)*(1.0-cos_theta));
    return static_cast<float>( (Re*Re/2) * P * (1 - P * sin_theta_2 + P * P));
}

float
ScatterSimulation::
photon_energy_after_Compton_scatter(const float cos_theta, const float energy)
{
    return static_cast<float>(energy/(1+(energy/511.0f)*(1-cos_theta)));   // For an arbitrary energy
}

float
ScatterSimulation::
photon_energy_after_Compton_scatter_511keV(const float cos_theta)
{
    return 511.f/(2.f-cos_theta); // for a given energy, energy := 511 keV
}

float
ScatterSimulation::
total_Compton_cross_section(const float energy)
{
    const double a= energy/511.0;
    const double l= log(1.0+2.0*a);
    const double sigma0= 6.65E-25;   // sigma0=8*pi*a*a/(3*m*m)
    return static_cast<float>( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) ); // Klein - Nishina formula = sigma / sigma0
}


float
ScatterSimulation::
total_Compton_cross_section_relative_to_511keV(const float energy)
{
    const double a= energy/511.0;
    static const double prefactor = 9.0/(-40 + 27*log(3.)); //Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9

    return //checked this in Mathematica
            static_cast<float>
            (prefactor*
             (((-4 - a*(16 + a*(18 + 2*a)))/square(1 + 2*a) +
               ((2 + (2 - a)*a)*log(1 + 2*a))/a)/square(a)
              ));
}

END_NAMESPACE_STIR
