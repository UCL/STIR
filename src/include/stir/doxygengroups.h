//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, IRSL
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

  In the future, we might use a hierarchy of namespaces, but we don't yet...
 */

/*!
\defgroup buildblock Basic building blocks
Library with things that are not not specific to reconstructions.
This includes multi-dimensional arrays, images, image processors, 
projection data,...
*/
/*!
\defgroup recon_buildblock Reconstruction building blocks
Library with 'general' reconstruction building blocks
*/
/*!
\defgroup LogLikBased_buildblock Reconstruction building blocks for loglikelihood based algorithms
Library with additional building blocks used for algorithms which
are similar to EM.
*/
/*!
\defgroup display Display functions
Library for displaying of images
*/
/*!
\defgroup para Parallel library 
*/
/*!
\defgroup test Tests of the basic building blocks
*/
/*!
\defgroup recontest Tests of reconstruction building blocks
*/
/*!
\defgroup utilities Utility programmes
*/
/*!
\defgroup reconstructors Reconstruction classes
*/
/*!
\defgroup OSMAPOSL Implementation of the OSMAP One-Step-Late reconstruction algorithm
*/
