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
/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::SingleScatterLikelihoodAndGradient

  \author Ludovica Brusaferri
*/

#ifndef __stir_scatter_SingleScatterLikelihoodAndGradient_H__
#define __stir_scatter_SingleScatterLikelihoodAndGradient_H__

#include "stir/Succeeded.h"
#include "stir/scatter/ScatterSimulation.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/RegisteredParsingObject.h"


START_NAMESPACE_STIR

class SingleScatterLikelihoodAndGradient : public
        RegisteredParsingObject<
        SingleScatterLikelihoodAndGradient,
        SingleScatterSimulation,
        SingleScatterSimulation>
{
private:
    typedef RegisteredParsingObject<
    SingleScatterLikelihoodAndGradient,
    SingleScatterSimulation,
    SingleScatterSimulation> base_type;
public:



    //! Name which will be used when parsing a ScatterSimulation object
    static const char * const registered_name;

    //! Default constructor
    SingleScatterLikelihoodAndGradient();

    //! Constructor with initialisation from parameter file
    explicit
    SingleScatterLikelihoodAndGradient(const std::string& parameter_filename);

    virtual ~SingleScatterLikelihoodAndGradient();



    double L_G_function(ProjData& data,VoxelsOnCartesianGrid<float>& gradient_image, const float rescale , bool isgradient = true);



    //void initialise(const std::string& parameter_filename);



    protected:

    void
    line_contribution(VoxelsOnCartesianGrid<float>& gradient_image,const float scale,
                                  const CartesianCoordinate3D<float>& scatter_point,
                                  const CartesianCoordinate3D<float>& detector_coord,
        						  const float C);

    void
    s_contribution(VoxelsOnCartesianGrid<float>& gradient_image,
        		const CartesianCoordinate3D<float>& scatter_point,
        						  const float D);
    float
    L_G_for_one_scatter_point(VoxelsOnCartesianGrid<float>& gradient,
             const std::size_t scatter_point_num,
             const unsigned det_num_A,
             const unsigned det_num_B, bool isgradient);

    double L_G_estimate(VoxelsOnCartesianGrid<float>& gradient_image_bin,const Bin bin, bool isgradient);

    /*virtual void
    actual_L_G_estimate(VoxelsOnCartesianGrid<float>& gradient_image_bin,
    		double& scatter_ratio_singles,
    			const unsigned det_num_A,
                const unsigned det_num_B, bool isgradient);*/


    double L_G_for_view_segment_number(ProjData&data,VoxelsOnCartesianGrid<float>& gradient_image,const ViewSegmentNumbers& vs_num, const float rescale, bool isgradient);

    double L_G_for_viewgram(Viewgram<float>& viewgram,Viewgram<float>& v_est,VoxelsOnCartesianGrid<float>& gradient_image,const float rescale, bool isgradient);



};

END_NAMESPACE_STIR

#endif
