/*
    Copyright (C) 2023 University College London
    Copyright (C) 2022 National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::LOR and derived classes

  \author Kris Thielemans
  \author Daniel Deidda
*/

%shared_ptr(stir::LOR<float>);
%shared_ptr(stir::LORAs2Points<float>);
%shared_ptr(stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>);
%shared_ptr(stir::LORInAxialAndSinogramCoordinates<float>);
%shared_ptr(stir::PointOnCylinder<float>);
%shared_ptr(stir::LORInCylinderCoordinates<float>);

#if 0
 // TODO enable this in STIR version 6 (breaks backwards compatibility
%attributeref(stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>, float, z1);
%attributeref(stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>, float, z2);
%attributeref(stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>, float, beta);
%attributeref(stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>, float, phi);
#else
%ignore *::z1() const;
%ignore *::z2() const;
%ignore *::beta() const;
%ignore *::phi() const;
#endif
%ignore *::check_state;

%attributeref(stir::PointOnCylinder<float>, float, z);
%attributeref(stir::PointOnCylinder<float>, float, psi);

%include "stir/LORCoordinates.h"

%template(LOR) stir::LOR<float>;
%template(LORInAxialAndNoArcCorrSinogramCoordinates) stir::LORInAxialAndNoArcCorrSinogramCoordinates<float>;
%template(LORAs2Points) stir::LORAs2Points<float>;
%template(LORInAxialAndSinogramCoordinates) stir::LORInAxialAndSinogramCoordinates<float>;
%template(PointOnCylinder) stir::PointOnCylinder<float>;
%template(LORInCylinderCoordinates) stir::LORInCylinderCoordinates<float>;
