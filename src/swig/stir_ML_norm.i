/*
    Copyright (C) 2024 University College London
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::ML_estimate_component_based_normalisation etc

  \author Kris Thielemans
  \author Markus Jehl

*/

%shared_ptr(stir::FanProjData);
%shared_ptr(stir::GeoData3D);
%ignore operator<<;
%ignore operator>>;
%ignore stir::DetPairData::operator()(const int a, const int b) const;
%ignore stir::DetPairData3D::operator()(const int a, const int b) const;
%ignore stir::FanProjData::operator()(const int, const int, const int, const int) const;
%ignore stir::GeoData3D::operator()(const int, const int, const int, const int) const;
%include "stir/ML_norm.h"
%include "stir/recon_buildblock/ML_estimate_component_based_normalisation.h"
