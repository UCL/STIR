//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup motion
  \brief Functions to interpolate sinograms

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/VectorWithOffset.h"
#include "stir/SegmentByView.h"
#include "stir/Bin.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/Coordinate4D.h"
#include "stir/LORCoordinates.h"
#include "stir/DetectionPositionPair.h"

START_NAMESPACE_STIR

static
Succeeded
get_transformed_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& out_lor,
		    const RigidObject3DTransformation& transformation,
		    const Bin& bin,
		    const ProjDataInfo& in_proj_data_info)
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  in_proj_data_info.get_LOR(lor, bin);
  LORAs2Points<float> lor_as_points;
  lor.get_intersections_with_cylinder(lor_as_points, lor.radius());
  // TODO origin
  // currently, the origin used for  proj_data_info is in the centre of the scanner,
  // while for standard images it is in the centre of the first ring.
  // This is pretty horrible though, as the transform_point function has no clue 
  // where the origin is
  // Note that the present shift will make this version compatible with the 
  // version above, as find_bin_given_cartesian_coordinates_of_detection
  // also uses an origin in the centre of the first ring
  const float z_shift = 
    (in_proj_data_info.get_scanner_ptr()->get_num_rings()-1)/2.F *
    in_proj_data_info.get_scanner_ptr()->get_ring_spacing();
  lor_as_points.p1().z() += z_shift;
  lor_as_points.p2().z() += z_shift;
  LORAs2Points<float> 
    transformed_lor_as_points(transformation.transform_point(lor_as_points.p1()),
			      transformation.transform_point(lor_as_points.p2()));
  transformed_lor_as_points.p1().z() -= z_shift;
  transformed_lor_as_points.p2().z() -= z_shift;
  return transformed_lor_as_points.change_representation(out_lor, lor.radius());
}

static
Coordinate4D<float>
lor_to_coords(const LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor)
{
  Coordinate4D<float> coord;
  coord[1] = lor.z2()-lor.z1();
  coord[2] = (lor.z2()+lor.z1())/2;
  coord[3] = lor.phi();
  coord[4] = lor.beta();
  return coord;
}

static
Coordinate4D<int>
bin_to_coords(const Bin& bin)
{
  Coordinate4D<int> coord;
  coord[1] = bin.segment_num();
  coord[2] = bin.axial_pos_num();
  coord[3] = bin.view_num();
  coord[4] = bin.tangential_pos_num();
  return coord;
}


static
Bin
coords_to_bin(const Coordinate4D<int>& coord)
{
  Bin bin;
  bin.segment_num() = coord[1];
  bin.axial_pos_num() = coord[2];
  bin.view_num() = coord[3];
  bin.tangential_pos_num() = coord[4];
  return bin;
}

inline int sign(float x)
{ return x>=0 ? 1 : -1; }

inline 
Coordinate4D<int> sign (const Coordinate4D<float>& c)
{
  return Coordinate4D<int> (sign(c[1]),sign(c[2]),sign(c[3]),sign(c[4]));
}

inline 
Coordinate4D<float> abs (const Coordinate4D<float>& c)
{
  return Coordinate4D<float> (fabs(c[1]),fabs(c[2]),fabs(c[3]),fabs(c[4]));
}

inline
void add_to_bin(VectorWithOffset<shared_ptr<SegmentByView<float> > > & segments,
		const Bin& bin,
		const float value)
{
  (*segments[bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] += 
    value;
}

static
bool
is_in_range(const Bin& bin, const ProjDataInfo& proj_data_info)
{
  if (bin.segment_num()>= proj_data_info.get_min_segment_num()
      && bin.segment_num()<= proj_data_info.get_max_segment_num()
      && bin.tangential_pos_num()>= proj_data_info.get_min_tangential_pos_num()
      && bin.tangential_pos_num()<= proj_data_info.get_max_tangential_pos_num()
      && bin.axial_pos_num()>=proj_data_info.get_min_axial_pos_num(bin.segment_num())
      && bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num())
      ) 
    {
      assert(bin.view_num()>=proj_data_info.get_min_view_num());
      assert(bin.view_num()<=proj_data_info.get_max_view_num());
      return true;
    }
  else
    return false;
}

static
Bin
standardise(const Bin& in_bin, const ProjDataInfo& proj_data_info)
{
  Bin bin = in_bin;
  bool swap_direction = false;
  if (bin.view_num()< proj_data_info.get_min_view_num())
    {
      swap_direction = true;
      bin.view_num()+=proj_data_info.get_num_views();
    }
  else if (bin.view_num() > proj_data_info.get_max_view_num())
    {
      swap_direction = true;
      bin.view_num()-=proj_data_info.get_num_views();
    }
  assert(bin.view_num()>=proj_data_info.get_min_view_num());
  assert(bin.view_num()<=proj_data_info.get_max_view_num());

  if (swap_direction)
    {
    bin.tangential_pos_num() *= -1;
    bin.segment_num() *= -1;
    }
  return bin;
}


static
Succeeded 
find_sampling(Coordinate4D<float>& sampling, const Bin& bin, const ProjDataInfo& proj_data_info)
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> tmp_lor;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  proj_data_info.get_LOR(lor, bin);
  for (int d=1; d<=4; ++d)
    {
      Coordinate4D<int> neighbour_coords = bin_to_coords(bin);
      neighbour_coords[d]+=1;
      const Bin neighbour = standardise(coords_to_bin(neighbour_coords),
					proj_data_info);
      if (!is_in_range(neighbour, proj_data_info))
	return Succeeded::no;
      proj_data_info.get_LOR(tmp_lor, neighbour);
      sampling[d] = (lor_to_coords(tmp_lor) - lor_to_coords(lor))[d];
    }
  return Succeeded::yes;
}

static
void
bin_interpolate(VectorWithOffset<shared_ptr<SegmentByView<float> > > & seg_ptr,
		const LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor,
		const ProjDataInfo& out_proj_data_info,
		const ProjDataInfo& proj_data_info,
		const float value)
{
  Coordinate4D<float> sampling;
  //warning assuming uniform sampling to avoid problem of neighbour dropping of at the edge
  if (find_sampling(sampling, Bin(0,0,0,0), proj_data_info)!= Succeeded::yes)
    error("error in finding sampling");

  const Bin central_bin = proj_data_info.get_bin(lor);
  if (central_bin.get_bin_value()<0)
    return;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> central_lor;
  proj_data_info.get_LOR(central_lor, central_bin);
  const Coordinate4D<float> central_lor_coords = lor_to_coords(central_lor);
  const Coordinate4D<float> lor_coords = lor_to_coords(lor);
  const Coordinate4D<int> central_bin_coords = bin_to_coords(central_bin);
  const Coordinate4D<float> diff = (lor_coords - central_lor_coords)/ sampling;
  assert(diff[1]<=1.001 && diff[1]>=-1.001);
  assert(diff[2]<=.5001 && diff[2]>=-.5001);
  assert(diff[3]<=.5001 && diff[3]>=-.5001);
  assert(diff[4]<=.5001 && diff[4]>=-.5001);
  const Coordinate4D<int> bin_inc = sign(sampling);
  assert(bin_inc == Coordinate4D<int>(1,1,1,1));
  const Coordinate4D<int> inc = sign(diff);
  Coordinate4D<int> offset;

#if 0
  for (offset[1]=0; offset[1]!=2*inc[1]; offset[1]+=inc[1]) 
    for (offset[2]=0; offset[2]!=2*inc[2]; offset[2]+=inc[2]) 
      for (offset[3]=0; offset[3]!=2*inc[3]; offset[3]+=inc[3]) 
	for (offset[4]=0; offset[4]!=2*inc[4]; offset[4]+=inc[4]) 
#else
  for (offset[1]=-1; offset[1]!=2; offset[1]++) 
    for (offset[2]=-1; offset[2]!=2; offset[2]++) 
      for (offset[3]=-1; offset[3]!=2; offset[3]++) 
	for (offset[4]=-1; offset[4]!=2; offset[4]++) 
#endif
	  {
	    const Bin target_bin = 
	      standardise(coords_to_bin(central_bin_coords + offset), proj_data_info);
	    if (is_in_range(target_bin, proj_data_info))
	    {
	      
	      //const BasicCoordinate<4,float> float_offset(offset);
	      //Coordinate4D<float> weights = abs(diff - float_offset);
	      

	      LORInAxialAndNoArcCorrSinogramCoordinates<float> new_lor;
	      proj_data_info.get_LOR(new_lor, target_bin);
	      Coordinate4D<float> new_lor_coords = lor_to_coords(new_lor);
	      Coordinate4D<float> new_diff = (lor_coords - new_lor_coords)/ sampling;
	      if (fabs(new_diff[3])>2)
		{
		  new_lor_coords[1] *= -1;
		  new_lor_coords[3] += _PI*sign(new_diff[3]);
		  new_lor_coords[4]*=-1;
		  new_diff = (lor_coords - new_lor_coords)/ sampling;
		}
		  
#if 0
	      assert(new_diff[1]<=1.001 && new_diff[1]>=-1.001);
	      assert(new_diff[2]<=1.101 && new_diff[2]>=-1.101);
	      assert(new_diff[3]<=1.001 && new_diff[3]>=-1.001);
	      assert(new_diff[4]<=1.001 && new_diff[4]>=-1.001);
#endif
	      // assert(norm(new_diff - (diff - float_offset))<.001);
	      Coordinate4D<float> weights = abs(new_diff);
	      if (weights[4]>1 || weights[3]>1 || weights[2]>1 || weights[1]>1)
		continue;
	      const float weight = (1-weights[1])*(1-weights[2])*(1-weights[3])*(1-weights[4]);
	      if (weight<0.001)
		continue;
#if 0
	      const Bin out_bin = target_bin;
#else
#if 0
	      const Bin out_bin = out_proj_data_info.get_bin(new_lor);
	      if (out_bin.get_bin_value()<0)
		continue;
#else
	      DetectionPositionPair<> detection_positions;
	      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>
		(proj_data_info).
		get_det_pos_pair_for_bin(detection_positions, target_bin);
	      Bin out_bin;
	      if (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>
		  (out_proj_data_info).
		  get_bin_for_det_pos_pair(out_bin, detection_positions) == Succeeded::no)
		continue;
	      if (!is_in_range(out_bin, out_proj_data_info))
		continue;
#endif
#endif

	      add_to_bin(seg_ptr, out_bin, value*weight);
	    }

	  }
}

END_NAMESPACE_STIR
