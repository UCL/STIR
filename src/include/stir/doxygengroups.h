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
Library with things that are not not specific to reconstructions.
This includes multi-dimensional arrays, images, image processors, 
projection data,...
\todo Define more submodules in the doxygen documentation such that this
looks a lot neater.
*/
/*!
\defgroup Array Items relating to vectors and (multi-dimensional) arrays
\ingroup buildblock
*/
/*!
\defgroup Coordinate Items relating to coordinates
\ingroup buildblock
*/
/*!
\defgroup projdata Items related to projection data
\ingroup buildblock
Basic support for projection data. This is the term generally used in STIR
for data obtained by the scanner or immediate post-processing.
*/
/*!
\defgroup densitydata Items related to image data
\ingroup buildblock
Basic support for image (or discretised density) data. 
*/
/*!
\defgroup ImageProcessor Image processors
\ingroup densitydata
A hierarchy of classes for performing image processing. Mechanisms
for parsing are provided such that different image processors can
be selected at run-time.

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
\defgroup recon_buildblock Reconstruction building blocks
\ingroup STIR_library
Library with 'general' reconstruction building blocks
*/
/*!
\defgroup LogLikBased_buildblock Reconstruction building blocks for loglikelihood based algorithms
\ingroup recon_buildblock
Library with additional building blocks used for algorithms which
are similar to EM.
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

