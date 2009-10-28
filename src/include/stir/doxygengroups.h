//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

// define some doxygen groups and namespace
// This file does not contain any code

/*! \namespace stir
  \brief Namespace for the STIR library (and some/most of its applications)

  This namespace encompasses the whole
  library. All classes, functions and symbols are in this namespace.
  This has the effect that conflicts with any other library is
  impossible (except if that library uses the same namespace...).
 */

/*! \namespace stir::BSpline
  \brief Namespace for functions and classes for B-Spline interpolation in the STIR library
 */

/*! \namespace stir::detail
  \brief Namespace for the implementation details of the STIR library

  You should never have to call one of these routines.
*/

/*! \namespace stir::ecat
  \brief Namespace for the ECAT IO part of the STIR library (and some/most of its applications)

  This namespace contains all routines that are common to the ECAT6 and 
  ECAT7 format.
*/

/*! \namespace stir::ecat::ecat7
  \brief Namespace for the ECAT7 IO part of the STIR library (and some/most of its applications)

  This namespace is only non-empty when the HAVE_LLN_MATRIX preprocessor
  symbol is defined during compilation. 
  */

// have to define it here, otherwise doxygen ignores the \def command below
#define  HAVE_LLN_MATRIX

/*! \def HAVE_LLN_MATRIX
    \brief Preprocessor symbol that needs to be defined to enable ECAT7 support.
  You need to have the ecat matrix library developed originally at the 
  UCL of Louvain la Neuve. If the STIR Makefiles can find this 
  library, HAVE_LLN_MATRIX will be automatically defined for you.
  See the User's guide for instructions.
*/
/*! \namespace stir::ecat::ecat6
  \brief Namespace for the ECAT6 IO part of the STIR library (and some/most of its applications)
  */
/*!
\defgroup STIR STIR
All of STIR.
*/

/*!
\defgroup STIR_library STIR library
\ingroup STIR
The whole collection of libraries in STIR.
*/

/*!
\defgroup buildblock Basic building blocks
\ingroup STIR_library
Library with things that are not specific to reconstructions.
This includes multi-dimensional arrays, images, image processors, 
projection data,...
*/
/*!
\defgroup buildblock_detail Implementation details for buildblock 
\ingroup buildblock
*/
/*!
\defgroup Array Items relating to vectors and (multi-dimensional) arrays
\ingroup buildblock
*/
/*!
\defgroup Array_detail Implementation details used by Array classes
\ingroup Array
*/
/*!
\defgroup Array_IO Functions that implement IO for Array objects
\ingroup Array
*/
/*!
\defgroup Array_IO_detail Implementation details for functions that implement IO for Array objects
\ingroup Array_IO
*/
/*!
\defgroup Coordinate Items relating to coordinates
\ingroup buildblock
*/
/*!
\defgroup geometry Items related to simple geometric calculations
\ingroup buildblock
Functions to compute distances between lines etc.
*/
/*!
\defgroup projdata Items related to projection data
\ingroup buildblock
Basic support for projection data. This is the term generally used in STIR
for data obtained by the scanner or immediate post-processing.
*/
/*!
\defgroup LOR Items related to Line Of Responses (preliminary)
\ingroup projdata
Classes for LORs.
\warning Preliminary and likely to change in the next release
*/
/*!
\defgroup densitydata Items related to image data
\ingroup buildblock
Basic support for image (or discretised density) data. 
*/
/*!
\defgroup DataProcessor Data processors
\ingroup buildblock
A hierarchy of classes for performing data processing. Mechanisms
for parsing are provided such that different image processors can
be selected at run-time.
*/
/*!
\defgroup ImageProcessor Image processors
\ingroup densitydata
\see Doxygen group DataProcessor for other members!

A hierarchy of classes for performing image processing. Mechanisms
for parsing are provided such that different image processors can
be selected at run-time.

*/


/*!
\defgroup data_buildblock Acquisition data building blocks
\ingroup STIR_library
Library with building blocks for reading scan data
\todo move projection data etc in here
*/
/*! 
\defgroup singles_buildblock Singles rates etc
\ingroup data_buildblock
*/

/*!
\defgroup numerics Numerical algorithms
\ingroup STIR_library
*/
/*! 
\defgroup DFT Discrete Fourier transforms
\ingroup numerics
*/
/*! 
\defgroup BSpline Classes and functions for B-spline interpolation.
\ingroup numerics
*/
/*!
\defgroup IO Input/Output Library
\ingroup STIR_library
Library with classes and functions to read and write images and projection 
from/to file.
*/
/*!
\defgroup InterfileIO Interfile support in the IO library
\ingroup IO
*/
/*!
\defgroup ECAT ECAT6 and ECAT7 support in the IO library
\ingroup IO
*/


/*! 
\defgroup listmode Support classes for reading list mode data
\ingroup STIR_library
*/

/*! 
\defgroup Shape Classes for describing geometric shapes such as cylinders etc.
\ingroup STIR_library
*/

/*! 
\defgroup evaluation Classes for computing ROI values and other FOMs
   For image evaluation, it is often necessary to compute ROI values, 
   or other simple Figures of Merits (FOMs). These classes
   and functions allow you do to this directly in STIR. This is mainly
   useful for automation, as there is no nice graphical interface in STIR
   to draw ROIs etc.
\ingroup STIR_library
*/

/*!
\defgroup recon_buildblock Reconstruction building blocks
\ingroup STIR_library
Library with 'general' reconstruction building blocks
*/
/*!
\defgroup projection Projection building blocks
\ingroup recon_buildblock
Everything (?) related to projection matrices, forward and back projection.

In the context of image reconstruction, 'forward projection' means going from 
the image to an estimate of the (mean of the) data. This is because in 
SPECT and PET, the measurements can be seen to be approximations of line 
integrals through the object.

STIR keeps this terminology, even though it is unfortunate. (For instance, 
a stir::ProjMatrix is not a projection matrix in the mathematical sense.)
*/
/*!
\defgroup symmetries Symmetries building blocks
\ingroup projection
Usually, there are (geometric) symmetries between the image and the projection 
data. This means that various elements of the projection matrix will be equal.
The classes in this module convert this concept into code, such that projection
matrices need only be computed for the 'independent' bins.
*/
/*!
\defgroup normalisation Normalisation building blocks
\ingroup recon_buildblock
Everything related to BinNormalisation classes.

In PET 'normalisation' is used to describe a multiplicative calibration of
every detector-pair. More generally, it can be used to the process of 
'correcting' projection data by multiplying every bin with a factor.
*/
/*!
\defgroup GeneralisedObjectiveFunction Objective functions for iterative estimation of variables
\ingroup recon_buildblock
Everything related to objective functions, i.e. functions that need to
be 'optimised' in some way.
*/
/*!
\defgroup priors Priors and penalties for MAP
\ingroup GeneralisedObjectiveFunction
Everything related to priors, which are used for MAP-type (also knows as
'penalised') reconstructions.
*/
/*!
\defgroup distributable distributable building blocks
\ingroup recon_buildblock
Classes and functions that are used to make a common interface for the serial
and parallel implementation of the reconstruction algorithms.
*/


/*!
\defgroup reconstructors Reconstruction classes
\ingroup STIR_library
*/
/*!
\defgroup OSMAPOSL OSMAPOSL
\ingroup reconstructors
Implementation of the OSMAP One-Step-Late reconstruction algorithm
*/
/*!
\defgroup OSSPS OSSPS
\ingroup reconstructors
Implementation of the OS Separable Paraboloidal Surrogate reconstruction algorithm
*/
/*!
\defgroup FBP2D FBP2D
\ingroup reconstructors
Implementation of the 2D Filtered Back Projection algorithm
*/
/*!
\defgroup FBP3DRP FBP3DRP
\ingroup reconstructors
Implementation of the 3D Reprojection Filtered Back Projection algorithm
*/

/*!
\defgroup modelling Kinetic modelling building blocks
\ingroup STIR_library
building blocks for kinetic modelling
*/
/*!
\defgroup scatter Scatter estimation building blocks
\ingroup STIR_library
building blocks for scatter estimation
*/

/*!
\defgroup display Display functions
\ingroup STIR_library
Library for displaying of images
*/
/*!
\defgroup para Parallel library 
\ingroup STIR_library
*/

/*!
\defgroup alltest Test code for STIR
\ingroup STIR
*/
/*!
\defgroup test Tests of the basic building blocks
\ingroup alltest
*/
/*!
\defgroup recontest Tests of reconstruction building blocks
\ingroup alltest
*/
/*!
\defgroup numerics_test Tests of numeric building blocks
\ingroup alltest
*/




/*!
\defgroup main_programs Executables
\ingroup STIR
Almost all programs that can be executed by the user.
*/
/*!
\defgroup utilities Utility programs
\ingroup main_programs
*/
/*!
\defgroup listmode_utilities Utility programs for list mode data
\ingroup utilities
*/
/*!
\defgroup ECAT_utilities ECAT6 and ECAT7 utilities
\ingroup utilities
Includes conversion programs etc.
*/



/*!
\defgroup examples Example files
\ingroup STIR
Some examples files to illustrate some basic coding in STIR.
*/
