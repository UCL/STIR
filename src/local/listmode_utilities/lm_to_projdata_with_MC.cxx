//
// $Id$
//
/*!
  \file 
  \ingroup utilities

  \brief Program to bin listmode data to 3d sinograms
 
  \author Kris Thielemans
  \author Sanida Mustafovic
  
  $Date$
  $Revision $
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

/* Possible compilation switches:
  
HIDACREBINNER: 
  Enable code specific for the HiDAC
INCLUDE_NORMALISATION_FACTORS: 
  Enable code to include normalisation factors while rebinning.
  Currently only available for the HiDAC.
USE_SegmentByView
  Currently our ProjData classes store segments as floats, which is a waste of
  memory and time for simple binning of listmode data. This should be
  remedied at some point by having member template functions to allow different
  data types in ProjData et al.
  Currently we work (somewhat tediously) around this problem by using Array classes directly.
  If you want to use the Segment classes (safer and cleaner)
  #define USE_SegmentByView
*/   
#define USE_SegmentByView

//#define HIDACREBINNER   
//#define INCLUDE_NORMALISATION_FACTORS

// set elem_type to what you want to use for the sinogram elements
// we need a signed type, as randoms can be subtracted. However, signed char could do.

#if defined(USE_SegmentByView) || defined(INCLUDE_NORMALISATION_FACTORS) 
   typedef float elem_type;
#  define OUTPUTNumericType NumericType::FLOAT
#else
   typedef short elem_type;
#  define OUTPUTNumericType NumericType::SHORT
#endif


#include "stir/utilities.h"
#ifdef HIDACREBINNER
#include "local/stir/QHidac/lm_qhidac.h"
#include "stir/ProjDataInfoCylindrical.h"
#else
#include "local/stir/listmode/lm.h"
#include "local/stir/listmode/CListModeData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#endif
#include "stir/Scanner.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInterfile.h"

#include "stir/ParsingObject.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/is_null_ptr.h"

#include "stir/CPUTimer.h"
#include "local/stir/listmode/LmToProjDataWithMC.h"
//#include "local/stir/listmode/LmToProjData.h"


#include <fstream>
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::ifstream;
using std::iostream;
using std::ofstream;
using std::streampos;
using std::cerr;
using std::cout;
using std::flush;
using std::endl;
using std::min;
using std::max;
#endif

/*
 Here's a sample .par file
\verbatim
LmToProjDataWithMC Parameters := 

input file := listmode data
output filename := precorrected_MC_data.s
  
Bin Normalisation type:= From ECAT7
  Bin Normalisation From ECAT7:=
  normalisation_ECAT7_filename:= norm_filename.n
End Bin Normalisation From ECAT7:=

  store_prompts:=
  delayed_increment:=
  do_time_frame := 
  start_time:=
  end_time:=
  num_segments_in_memory:


 
*/



USING_NAMESPACE_STIR


#ifdef USE_SegmentByView
#include "stir/SegmentByView.h"
typedef SegmentByView<elem_type> segment_type;
#else
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
typedef Array<3,elem_type> segment_type;
#endif



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  
  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " [par_file]\n";
    exit(EXIT_FAILURE);
  }
  LmToProjDataWithMC application(argc==2 ? argv[1] : 0);
  application.compute();

  return EXIT_SUCCESS;
}
