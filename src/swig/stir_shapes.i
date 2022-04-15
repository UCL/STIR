/*
    Copyright (C) 2018 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Shape3D hierarchy

  \author Kris Thielemans
*/

%shared_ptr(stir::Shape3D)
%shared_ptr(stir::Shape3DWithOrientation)
%shared_ptr(stir::RegisteredParsingObject<stir::Ellipsoid, stir::Shape3D, stir::Shape3DWithOrientation>)
%shared_ptr(stir::Ellipsoid)
%shared_ptr(stir::RegisteredParsingObject<stir::EllipsoidalCylinder, stir::Shape3D, stir::Shape3DWithOrientation>)
%shared_ptr(stir::EllipsoidalCylinder)
%shared_ptr(stir::RegisteredParsingObject<stir::Box3D, stir::Shape3D, stir::Shape3DWithOrientation>)
%shared_ptr(stir::Box3D)

%include "stir/Shape/Shape3D.h"
%include "stir/Shape/Shape3DWithOrientation.h"
%template(RPEllipsoid) stir::RegisteredParsingObject<stir::Ellipsoid, stir::Shape3D, stir::Shape3DWithOrientation>;
%template(RPEllipsoidalCylinder) stir::RegisteredParsingObject<stir::EllipsoidalCylinder, stir::Shape3D, stir::Shape3DWithOrientation>;
%template(RPBox3D) stir::RegisteredParsingObject<stir::Box3D, stir::Shape3D, stir::Shape3DWithOrientation>;
%include "stir/Shape/Ellipsoid.h"
%include "stir/Shape/EllipsoidalCylinder.h"
%include "stir/Shape/Box3D.h"
