//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementation of classes CListEventECAT962w for listmode events for the 
   ECAT 962 (aka Exact HR+).
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 1998- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

/*
   Warning:
   this code makes use of an implementation dependent feature:
   bit shifting negative ints to the right.
    -1 >> 1 should be -1
    -2 >> 1 should be -1
   This is ok on SUNs (gcc, but probably SUNs cc as well), Parsytec (gcc),
   Pentium (gcc, VC++) and probably every other system which uses
   the 2-complement convention.

  TODO insert assert
*/

#include "local/stir/listmode/CListRecordECAT962.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/ByteOrder.h"
#include "stir/Succeeded.h"

#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::swap;
using std::streamsize;
using std::streampos;
#endif

START_NAMESPACE_STIR

// static members

shared_ptr<Scanner>
CListRecordECAT962::
scanner_sptr =
  new Scanner(Scanner::E962);

shared_ptr<ProjDataInfoCylindricalNoArcCorr>
CListRecordECAT962::
uncompressed_proj_data_info_sptr =
   dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
   ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                 1, scanner_sptr->get_num_rings()-1,
                                 scanner_sptr->get_num_detectors_per_ring()/2,
                                 scanner_sptr->get_default_num_arccorrected_bins
(),
                                 false));

const int CListEventDataECAT962 ::num_views = 288;


/* Go from sinograms to detectors.

   TODO remove duplication of ProjDataInfoCylindricalNoArcCorr::
   get_det_num_pair_for_view_tangential_pos_num().

   Because sinograms are not arc-corrected, bin number corresponds
   to an angle as well. Before interleaving (see LM.h) we have that
   det_angle_a = LOR_angle + bin_angle
   det_angle_b = LOR_angle + (Pi - bin_angle)

   (Hint: understand this first at LOR_angle=0, then realise that
    other LOR_angles follow just by rotation)

   Code gets slightly intricate because:
   - angles have to be defined modulo 2 Pi (so 2*num_views)
   - interleaving
 */
static void sinogram_to_detectors(
			   int& det_num_a, int& det_num_b,
			   const int bin, const int view,
			   const int num_views)
{
  /*
     this uses code from CTI
     Note for implementation: avoid using % with negative numbers
     so add 2*nv before doing modulo 2*nv)
  */

#define ve_to_det1( e, v, nv )  \
     ( ( v + ( e >> 1 ) + 2 * nv ) % (2 * nv ) )
#define ve_to_det2( e, v, nv )  \
     ( ( v - ( ( e + 1 ) >> 1 ) + 3 * nv ) % (2 * nv ) )

  det_num_a = ve_to_det1( bin, view, num_views );
  det_num_b = ve_to_det2( bin, view, num_views );
}

/* TODO remove duplication of ProjDataInfoCylindricalNoArcCorr::
get_view_tangential_pos_num_for_det_num_pair*/

static int detectors_to_sinogram(
			   const int det_num_a, const int det_num_b,
			   int& bin, int& view,
			   const int num_views)
{
  int swap_detectors;

  /*
     Note for implementation: avoid using % with negative numbers
     so add 2*nv before doing modulo 2*nv

     This somewhat obscure formula was obtained by inverting the CTI code above
 */

  bin = (det_num_a - det_num_b +  3*num_views) % (2* num_views);
  view = (det_num_a - (bin >> 1) +  2*num_views) % (2* num_views);

  /* Now adjust ranges for view, bin.
     The next lines go only wrong in the singular (and irrelevant) case
     det_num_a == det_num_b (when bin == num_views == 2*num_views - bin)

     We use the combinations of the following 'symmetries' of
     sinogram_to_detectors():
     (bin, view) == (bin+2*num_views, view + num_views)
                 == (-bin, view + num_views)
     Using the latter interchanges det_num_a and det_num_b, and this leaves
     the LOR the same the 2D case. However, in 3D this interchanges the rings
     as well. So, we keep track of this in swap_detectors, and return its final
     value (see LM.h).
     */
  if (view <  num_views)
    {
      if (bin >=  num_views)
      {
	bin = 2* num_views - bin;
	swap_detectors = 1;
      }
      else
      {
        swap_detectors = 0;
      }
    }
  else
    {
      view -= num_views;
      if (bin >=  num_views)
      {
	bin -= 2* num_views;
        swap_detectors = 0;
      }
      else
      {
	bin *= -1;
	swap_detectors = 1;
      }
    }

  return swap_detectors;
}

/*	Global Definitions */
const int  MAXPROJBIN = 512;
/* data for the 962 scanner */
const int CRYSTALRINGSPERDETECTOR = 8;
//TODO NK check
void
CListEventDataECAT962::
get_sinogram_and_ring_coordinates(
		   int& view_num, int& tangential_pos_num, int& ring_a, int& ring_b) const
{
  const int NumProjBins = MAXPROJBIN;
  const int NumProjBinsBy2 = MAXPROJBIN / 2;

  view_num = view;
  tangential_pos_num = bin;
  /* KT 31/05/98 use >= in comparison now */
  if ( tangential_pos_num >= NumProjBinsBy2 )
      tangential_pos_num -= NumProjBins ;

  ring_a = ( (block_A_ring_bit0 + 2*block_A_ring_bit1) 
	     * CRYSTALRINGSPERDETECTOR ) +  block_A_detector ;
  ring_b = ( (block_B_ring_bit0 + 2*block_B_ring_bit1)
	     * CRYSTALRINGSPERDETECTOR ) +  block_B_detector ;
}

void 
CListEventDataECAT962::
set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const int ring_a, const int ring_b)
{
  const int NumProjBins = MAXPROJBIN;
  type = 0;
  const unsigned int block_A_ring     = ring_a / CRYSTALRINGSPERDETECTOR;
  block_A_detector = ring_a % CRYSTALRINGSPERDETECTOR;
  const unsigned int block_B_ring     = ring_b / CRYSTALRINGSPERDETECTOR;
  block_B_detector = ring_b % CRYSTALRINGSPERDETECTOR;

  assert(block_A_ring<4);
  block_A_ring_bit0 = block_A_ring | 0x1;
  block_A_ring_bit1 = block_A_ring/2;
  assert(block_B_ring<4);
  block_B_ring_bit0 = block_B_ring | 0x1;
  block_B_ring_bit1 = block_B_ring/2;
  
  bin = tangential_pos_num < 0 ? tangential_pos_num + NumProjBins : tangential_pos_num;
  view = view_num;
}


void 
CListEventDataECAT962::
get_detectors(
		   int& det_num_a, int& det_num_b, int& ring_a, int& ring_b) const
{
  int tangential_pos_num;
  int view_num;
  get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);

  sinogram_to_detectors(det_num_a, det_num_b, tangential_pos_num, view_num, num_views);
}

void 
CListEventDataECAT962::
set_detectors(
			const int det_num_a, const int det_num_b,
			const int ring_a, const int ring_b)
{
  int tangential_pos_num;
  int view_num;
  int swap_detectors =
    detectors_to_sinogram(det_num_a, det_num_b, tangential_pos_num, view_num, num_views);

  if (swap_detectors != 1)
  {
    set_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);
  }
  else
  {
     set_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_b, ring_a);
  }
}

// TODO maybe move to ProjDataInfoCylindricalNoArcCorr
static void
sinogram_coordinates_to_bin(Bin& bin, const int view_num, const int tang_pos_num, 
			const int ring_a, const int ring_b,
			const ProjDataInfoCylindrical& proj_data_info)
{
  if (proj_data_info.get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_a, ring_b) ==
      Succeeded::no)
    {
      bin.set_bin_value(-1);
      return;
    }
  bin.set_bin_value(1);
  bin.view_num() = view_num / proj_data_info.get_view_mashing_factor();  
  bin.tangential_pos_num() = tang_pos_num;
}

void 
CListEventDataECAT962::
get_bin(Bin& bin, const ProjDataInfoCylindrical& proj_data_info) const
{
  int tangential_pos_num;
  int view_num;
  int ring_a;
  int ring_b;
  get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);
  sinogram_coordinates_to_bin(bin, view_num, tangential_pos_num, ring_a, ring_b, proj_data_info);
}

void 
CListRecordECAT962::
get_uncompressed_bin(Bin& bin) const
{
  int ring_a;
  int ring_b;
  event_data.get_sinogram_and_ring_coordinates(bin.view_num(), bin.tangential_pos_num(), ring_a, ring_b);
  uncompressed_proj_data_info_sptr->
    get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), 
					    ring_a, ring_b);
}  




END_NAMESPACE_STIR
