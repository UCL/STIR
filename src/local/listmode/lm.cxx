//
// $Id$
//
/*
  version 0.4 of preliminary code to handle CTI listmode data and sinograms.

  Kris Thielemans,
  31/05/98

  see LM.h for more information


   Warning:
   this code makes use of an implementation dependent feature:
   bit shifting negative ints to the right.
    -1 >> 1 should be -1
    -2 >> 1 should be -1
   This is ok on SUNs (gcc, but probably SUNs cc as well), Parsytec (gcc),
   Pentium (gcc, VC++) and probably every other system which uses
   the 2-complement convention.

*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/lm.h"
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

// TODO this is appropriate for the 966 only
const int CListEvent ::num_views = 288;

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
/* data for the 966 scanner */
const int CRYSTALRINGSPERDETECTOR = 8;

void
CListEvent::
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

  ring_a = ( block_A_ring * CRYSTALRINGSPERDETECTOR ) +  block_A_detector ;
  ring_b = ( block_B_ring * CRYSTALRINGSPERDETECTOR ) +  block_B_detector ;
}

void 
CListEvent::
set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const int ring_a, const int ring_b)
{
  const int NumProjBins = MAXPROJBIN;
  type = 0;
  block_A_ring     = ring_a / CRYSTALRINGSPERDETECTOR;
  block_A_detector = ring_a % CRYSTALRINGSPERDETECTOR;
  block_B_ring     = ring_b / CRYSTALRINGSPERDETECTOR;
  block_B_detector = ring_b % CRYSTALRINGSPERDETECTOR;

  bin = tangential_pos_num < 0 ? tangential_pos_num + NumProjBins : tangential_pos_num;
  view = view_num;
}


void 
CListEvent::
get_detectors(
		   int& det_num_a, int& det_num_b, int& ring_a, int& ring_b) const
{
  int tangential_pos_num;
  int view_num;
  get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);

  sinogram_to_detectors(det_num_a, det_num_b, tangential_pos_num, view_num, num_views);
}

void 
CListEvent::
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
CListEvent::
get_bin(Bin& bin, const ProjDataInfoCylindrical& proj_data_info) const
{
  int tangential_pos_num;
  int view_num;
  int ring_a;
  int ring_b;
  get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);
  sinogram_coordinates_to_bin(bin, view_num, tangential_pos_num, ring_a, ring_b, proj_data_info);
}



#if 0
int get_next_event(istream&in, CListRecord& event)
{
  
  in.read(reinterpret_cast<char *>(&event), sizeof(event));
  
  if (in.good())
    return 1;
  if (in.eof())
    return 0; 
  else
  { error("Error after reading from stream in get_next_event\n"); }
  /* Silly statement to satisfy VC++, but we never get here */
  return 0;
  
}
#else
// this will skip last event in file
int get_next_event(istream&in, CListRecord& event)
{
  // TODO this is appropriate only for 966
#ifdef STIRByteOrderIsBigEndian
  assert(ByteOrder::get_native_order() == ByteOrder::big_endian);
#else
  assert(ByteOrder::get_native_order() == ByteOrder::little_endian);
#endif
  
  const unsigned int buf_size = 100000;
  static CListRecord buffer[buf_size];
  static unsigned int current_pos = buf_size;
  static streamsize num_events_in_buffer = 0;
  static streampos stream_position  = 0;
  if (current_pos == buf_size || stream_position != in.tellg())// check if user reset the stream position, if so, reinitialise buffer
  {
    //cerr << "Reading from listmode file \n";
    // read some more data
    in.read(reinterpret_cast<char *>(buffer), sizeof(event)*buf_size);
    current_pos=0;
    if (in.eof())
    {
      num_events_in_buffer = in.gcount();
    }
    else
    {
      if (!in.good())
      { error("Error after reading from stream in get_next_event\n"); }
      num_events_in_buffer = buf_size;
      assert(buf_size*sizeof(event)==in.gcount());
    }
    stream_position = in.tellg();
    
  }
  
  if (current_pos != static_cast<unsigned int>(num_events_in_buffer))
  {
    event = buffer[current_pos++];
#if (defined(STIRByteOrderIsBigEndian) && !defined(STIRListmodeFileFormatIsBigEndian)) \
    || (defined(STIRByteOrderIsLittleEndian) && defined(STIRListmodeFileFormatIsBigEndian)) 
    ByteOrder::swap_order(event);
#endif
    return 1;
  }
  else
  {
    return 0;
  }
}

#endif


END_NAMESPACE_STIR
