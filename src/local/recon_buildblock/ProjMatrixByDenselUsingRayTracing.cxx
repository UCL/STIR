//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByDenselUsingRayTracing

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "local/stir/recon_buildblock/DataSymmetriesForDensels_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include <algorithm>
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::max;
using std::min;
#endif

START_NAMESPACE_STIR


const char * const 
ProjMatrixByDenselUsingRayTracing::registered_name =
  "Ray Tracing";

ProjMatrixByDenselUsingRayTracing::
ProjMatrixByDenselUsingRayTracing()
{
  set_defaults();
}

void 
ProjMatrixByDenselUsingRayTracing::initialise_keymap()
{
  parser.add_start_key("Ray Tracing Matrix Parameters");
  parser.add_stop_key("End Ray Tracing Matrix Parameters");
}


void
ProjMatrixByDenselUsingRayTracing::set_defaults()
{}

const DataSymmetriesForDensels*
ProjMatrixByDenselUsingRayTracing:: get_symmetries_ptr() const
{
  return  symmetries_ptr.get();
}

void
ProjMatrixByDenselUsingRayTracing::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByDenselUsingRayTracing initialised with a wrong type of DiscretisedDensity\n");

 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);

  symmetries_ptr = 
    new DataSymmetriesForDensels_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_ptr);
  const float sampling_distance_of_adjacent_LORs_xy =
    proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));
  
  if(sampling_distance_of_adjacent_LORs_xy > voxel_size.x() + 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy > voxel_size.y() + 1.E-3)
     warning("WARNING: ProjMatrixByDenselUsingRayTracing used for pixel size (in x,y) "
             "that is smaller than the densel size.\n"
             "This matrix will completely miss some voxels for some (or all) views.\n");
  if(sampling_distance_of_adjacent_LORs_xy < voxel_size.x() - 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy < voxel_size.y() - 1.E-3)
     warning("WARNING: ProjMatrixByDenselUsingRayTracing used for pixel size (in x,y) "
             "that is larger than the densel size.\n"
             "Backprojecting with this matrix will have artefacts at views 0 and 90 degrees.\n");

  xhalfsize = voxel_size.x()/2;
  yhalfsize = voxel_size.y()/2;
  zhalfsize = voxel_size.z()/2;
};

#if 0
/* this is used when 
   (tantheta==0 && sampling_distance_of_adjacent_LORs_z==2*voxel_size.z())
  it adds two  adjacents z with their half value
  */
static void 
add_adjacent_z(ProjMatrixElemsForOneDensel& probs);

/* Complicated business to add the same values at z+1
   while taking care that the (x,y,z) coordinates remain unique in the LOR.
  (If you copy the LOR somewhere else, you can simply use 
   ProjMatrixElemsForOneDensel::merge())
*/         
static void merge_zplus1(ProjMatrixElemsForOneDensel& probs);
#endif
static inline int sign(const float x) 
{ return x>=0 ? 1 : - 1; }


// for positive halfsizes, this is valid for 0<=phi<=Pi/2 && 0<theta<Pi/2
static inline float 
  projection_of_voxel_help(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const float ctheta, const float tantheta, 
                      float cphi, float sphi,
                      const float m, const float s)
{
  const float epsilon = 1.E-4F;
  if (fabs(sphi)<epsilon)
    sphi=sign(sphi)*epsilon;
  else if (fabs(cphi)<epsilon)
    cphi=sign(cphi)*epsilon;
  const float zs = zctr - m; 
  const float ys = yctr - s*cphi;
  const float xs = xctr + s*sphi;
  return
    max((-max((zs - zhalfsize)/tantheta, 
            max((ys - yhalfsize)/sphi, (xs - xhalfsize)/cphi)) +
         min((zs + zhalfsize)/tantheta, 
            min((ys + yhalfsize)/sphi, (xs + xhalfsize)/cphi)))/ctheta,
     0.F);
}

// for positive halfsizes, this is valid for 0<=phi<=Pi/2 && 0==theta
static inline float 
  projection2D_of_voxel_help(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      float cphi, float sphi,
                      const float m, const float s)
{
  const float epsilon = 1.E-4F;
  
  if (zhalfsize - fabs(zctr - m) <= 0)
    return 0.F;
#if 1
  if (fabs(sphi)<epsilon)
    sphi=sign(sphi)*epsilon;
  else if (fabs(cphi)<epsilon)
    cphi=sign(cphi)*epsilon;
#else
  // should work, but doesn't
  if (fabs(sphi)<epsilon)
    return (yhalfsize - fabs(yctr-s)) <= 0 ? 0 : 2*xhalfsize;
  if (fabs(cphi)<epsilon)
    return (xhalfsize - fabs(xctr-s)) <= 0 ? 0 : 2*yhalfsize;
#endif
  const float ys = yctr - s*cphi;
  const float xs = xctr + s*sphi;
  return
    max(-max((ys - yhalfsize)/sphi, (xs - xhalfsize)/cphi) +
         min((ys + yhalfsize)/sphi, (xs + xhalfsize)/cphi),
     0.F);
}


static inline float 
  projection_of_voxel(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const float ctheta, const float tantheta, 
                      const float cphi, const float sphi,
                      const float m, const float s)
{
  // if you want to relax the next assertion, replace yhalfsize with sign(sphi)*yhalfsize below
  //assert(sphi>0);
  return
    fabs(tantheta)<1.E-4 ?
       projection2D_of_voxel_help(xctr, yctr, zctr,
                                  sign(cphi)*xhalfsize, sign(sphi)*yhalfsize, zhalfsize,
                                  cphi, sphi,
                                  m, s)
                                  :
       projection_of_voxel_help(xctr, yctr, zctr,
                                  sign(cphi)*xhalfsize, sign(sphi)*yhalfsize, sign(tantheta)*zhalfsize,
                                  ctheta, tantheta,
                                  cphi, sphi,
                                  m, s);
}

static inline float 
  projection_of_voxel(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const Bin& bin, const ProjDataInfo& proj_data_info)
{
  const float tantheta = proj_data_info.get_tantheta(bin);
  const float costheta = 1/sqrt(1+square(tantheta));
  const float m = proj_data_info.get_t(bin)/costheta;
  // phi in KT's Mathematica conventions
  const float phi = proj_data_info.get_phi(bin) + _PI/2; 
  const float cphi = cos(phi);
  const float sphi = sin(phi);
  const float s = -proj_data_info.get_s(bin);
  
  return
    projection_of_voxel(xctr, yctr, zctr,
                        xhalfsize, yhalfsize, zhalfsize,
                        costheta, tantheta,
                        cphi, sphi,
                        m, s);
}

//////////////////////////////////////
void 
ProjMatrixByDenselUsingRayTracing::
calculate_proj_matrix_elems_for_one_densel(
                                        ProjMatrixElemsForOneDensel& probs) const
{
  const Densel densel = probs.get_densel();

  const float xctr = densel[3] * voxel_size.x() - origin.x();
  const float yctr = densel[2] * voxel_size.y() - origin.y();
  const float zctr = 
    (densel[1] - (min_index.z() + max_index.z())/2.F) * voxel_size.z() +
    origin.z();

  assert(probs.size() == 0);
#if 0     
  const float sampling_distance_of_adjacent_LORs_z =
    proj_data_info_ptr->get_sampling_in_t(bin)/costheta;
 

  // find number of LORs we have to take, such that we don't miss voxels
  // we have to subtract a tiny amount from the quotient, to avoid having too many LORs
  // solely due to numerical rounding errors
  const int num_lors_per_axial_pos = 
    static_cast<int>(ceil(sampling_distance_of_adjacent_LORs_z / voxel_size.z() - 1.E-3));

  assert(num_lors_per_axial_pos>0);
  // code below is currently restricted to 2 LORs
  assert(num_lors_per_axial_pos<=2);

  // merging code assumes integer multiple
  assert(fabs(sampling_distance_of_adjacent_LORs_z 
              - num_lors_per_axial_pos*voxel_size.z()) <= 1E-4);


  // find offset in z, taking into account if there are 1 or 2 LORs
  const float offset_in_z = 
    (num_lors_per_axial_pos == 1 || 
     tantheta == 0 ? 0.F : -sampling_distance_of_adjacent_LORs_z/4)
    - origin.z();
#endif

  // Now do a loop over the bins to store the non-zero projmatrix elements.
  //
  // The easiest is to just loop over all bins, but that's terribly slow.
  // First optimisation:
  //   I exit the loops over axial_pos_num and tangential_pos_num as soon as a 0 
  //   LOI is found after a non-zero one.
  //   This avoids computing lots of zeroes to the right of the non-zero range.
  //   This ASSUMES that the nonzero range of bins is connected.
  // Second optimisation:
  //   For the loop over tang_pos_num, I also keep track of which tang_pos_num gave the 
  //   first non-zero result for a view (stored in previous_min_tang_pos). The next 
  //   view will then start the tang_pos_num loop from 
  //   previous_min_tang_pos+previous_inc_min_tang_pos (where the increment is less than -2).
  //   This avoids computing lots of zeroes to the left of the non-zero range.
  //   There's a check that the increment was enough to the left, but there's still 
  //   some work to do there. See comments below.

  for (int seg = proj_data_info_ptr->get_min_segment_num(); seg <= proj_data_info_ptr->get_max_segment_num(); ++seg)
  {
    //if (seg!=0)
    //  error("ProjMatrixByDenselUsingRayTracing doesn't work for oblique segments yet\n");

    int previous_min_tang_pos = proj_data_info_ptr->get_min_tangential_pos_num();
    int previous_inc_min_tang_pos = -1;
    const int min_ax_pos = 
      proj_data_info_ptr->get_min_axial_pos_num(seg) +
      (densel[1]-max_index[1])/symmetries_ptr->get_num_planes_per_axial_pos(seg) - 1;
    const int max_ax_pos = 
      proj_data_info_ptr->get_max_axial_pos_num(seg) +
      (densel[1]-min_index[1])/symmetries_ptr->get_num_planes_per_axial_pos(seg) + 1;
    int previous_min_ax_pos = min_ax_pos;
    int previous_inc_min_ax_pos = -1;

    for (int view = proj_data_info_ptr->get_min_view_num(); view <= proj_data_info_ptr->get_max_view_num(); ++view)
    {
      bool found_nonzero_axial = false;
      int start_ax_pos = previous_min_ax_pos + previous_inc_min_ax_pos;
      int ax_pos_inc = -1;
      for (int ax_pos = start_ax_pos; ax_pos <= max_ax_pos; ax_pos+=ax_pos_inc)
      {
        if (ax_pos<min_ax_pos)
        {
          ax_pos_inc = 1;
          ax_pos=min_ax_pos;
          continue;
        }
        // else
        
        bool found_nonzero_tangential = false;
        int start_tang_pos = previous_min_tang_pos + previous_inc_min_tang_pos;
        
        // check if the increment wasn't too large (or not negative enough):
        // do this by looping until the current bin gives 0          
#if 1
        //std::cerr << "Start at tang_pos " << start_tang_pos 
        //          << " (" << seg << ',' << view << ',' << ax_pos << ')'<< std::endl;
        Bin bin(seg, view, ax_pos, 0);          
        int tang_pos_inc = -1;
        for (int tang_pos = start_tang_pos; tang_pos <= proj_data_info_ptr->get_max_tangential_pos_num(); tang_pos+=tang_pos_inc)
        {
          if (tang_pos<proj_data_info_ptr->get_min_tangential_pos_num())
          {
            tang_pos_inc = 1;
            tang_pos=proj_data_info_ptr->get_min_tangential_pos_num();
            continue;
          }
          // else
        
          bin.tangential_pos_num() = tang_pos;
          const float LOI = 
            projection_of_voxel(xctr, yctr, zctr,
                                xhalfsize, yhalfsize, zhalfsize,
                                bin, *proj_data_info_ptr);
          if (LOI > xhalfsize/1000.)
          {
            if (tang_pos_inc==-1)
            {
              // it's non-zero, check next bin to the left
              --previous_inc_min_tang_pos;
            }
            else
            {
              if (!found_nonzero_tangential) 
              {
                //std::cerr << "\tfirst tang_pos at " << tang_pos 
                //          << '(' << seg << ',' << view << ',' << ax_pos << ')'<< std::endl;
                previous_min_tang_pos = tang_pos;
                found_nonzero_tangential = true;
              }
              if (ax_pos_inc==+1)
              {
#ifdef NEWSCALE
                bin.set_bin_value(LOI); // normalise to mm
#else
                bin.set_bin_value(LOI/voxel_size.x()); // normalise to some kind of 'pixel units'
#endif
                probs.push_back(ProjMatrixElemsForOneDensel::value_type(bin));
              }
            }
          }
          else // the Pbv was zero
          {
            if (tang_pos_inc==-1)
            {
              tang_pos_inc=1;
            }
            else
            {
              if (found_nonzero_tangential)
              {
                // the first tang_pos where the result is zero again. So, all the next ones will be 0 as well.
                break;
              }
            }
          }
        } // end loop over tang_pos
        if (found_nonzero_tangential)
        {
          if (ax_pos_inc==-1)
          {
            // it's non-zero, check next bin to the left
            --previous_inc_min_ax_pos;
          }
          else
          {            
            if (!found_nonzero_axial)
            {
              previous_min_ax_pos = ax_pos;
              found_nonzero_axial = true;
            }
          }
        }
        else // all bins for this ax_pos were zero
        {
          if (ax_pos_inc==-1)
          {
            ax_pos_inc = 1;
          }
          else if (found_nonzero_axial)
          {            
            // the first ax_pos where the result is zero again. So, all the next ones will be 0 as well.
            break;
            // TODO potentially, the mechanism of using previous_min_ax_pos caused the 
            // ax_pos loop to miss to non-zero bins. See above
          }
        }
        
      }
    
#else
            while(true)
        {
          // if we're at the smallest bin, keep the increment
          if (start_tang_pos<=proj_data_info_ptr->get_min_tangential_pos_num())
          {
            start_tang_pos=proj_data_info_ptr->get_min_tangential_pos_num();
            break;
          }
          else
          {
            const float LOI = 
              projection_of_voxel(xctr, yctr, zctr,
                                  xhalfsize, yhalfsize, zhalfsize,
                                  Bin(seg, view, ax_pos, start_tang_pos), *proj_data_info_ptr);
            if (LOI > xhalfsize/1000.)
            {
              // it's non-zero, check next bin to the left
              --previous_inc_min_tang_pos;
              --start_tang_pos;
            }
            else
              break;
          }
        }
        if (start_tang_pos > proj_data_info_ptr->get_min_tangential_pos_num())
        {
          // the current bin was 0, so we don't have to redo it
          ++start_tang_pos;
        }
        //std::cerr << "Start at tang_pos " << start_tang_pos 
        //          << " (" << seg << ',' << view << ',' << ax_pos << ')'<< std::endl;
        for (int tang_pos = start_tang_pos; tang_pos <= proj_data_info_ptr->get_max_tangential_pos_num(); ++tang_pos)
        {
          Bin bin(seg, view, ax_pos, tang_pos);          
          const float LOI = 
            projection_of_voxel(xctr, yctr, zctr,
                                xhalfsize, yhalfsize, zhalfsize,
                                bin, *proj_data_info_ptr);
          if (LOI > xhalfsize/1000.)
          {
            if (!found_nonzero_tangential) 
            {
              //std::cerr << "\tfirst tang_pos at " << tang_pos 
              //          << '(' << seg << ',' << view << ',' << ax_pos << ')'<< std::endl;
              //XXXprevious_min_tang_pos = tang_pos;
            }
            found_nonzero_tangential = true;
#ifdef NEWSCALE
	    bin.set_bin_value(LOI); // normalise to mm
#else
	    bin.set_bin_value(LOI/voxel_size.x()); // normalise to some kind of 'pixel units'
#endif
            probs.push_back(ProjMatrixElemsForOneDensel::value_type(bin));
          }
          else if (found_nonzero_tangential)
          {
            // the first tang_pos where the result is zero again. So, all the next ones will be 0 as well.
            //XXXbreak;
          }
        } // end loop over tang_pos
        if (found_nonzero_axial)
        {
          if (!found_nonzero_tangential)
          {
            // the first ax_pos where the result is zero again. So, all the next ones will be 0 as well.
            //XXXbreak;
            // TODO potentially, the mechanism of using previous_min_tang_pos caused the 
            // tang_pos loop to miss to non-zero bins. This would occur if start_tang_pos was to
            // the 'right' of the nonzero range.
            // The only way I see to check this is to do a 
            // loop here to the left of the start_tang_pos;
          }
        }
        else
        {
          //if (found_nonzero_tangential) 
          //  std::cerr << "first ax_pos at " << ax_pos
          //            << '(' << seg << ',' << view << ')' << std::endl;
          found_nonzero_axial = found_nonzero_tangential;
        }
        
      }
#endif
      // next assert only possible when every voxel is detected for every seg,view
      assert(found_nonzero_axial);
    }
  }
  

#if 0           

  // now add on other LORs
  if ( num_lors_per_axial_pos>1)
  {      
    
    assert(num_lors_per_axial_pos==2);
    if (tantheta==0 ) 
    { 
      assert(Z1f == -origin.z()/voxel_size.z());
      add_adjacent_z(probs);
    }
    else
    { 
      // probs.merge( lor2 );   
      merge_zplus1(probs);
    }

  } // if( num_lors_per_axial_pos>1)
#endif
}
#if 0
// TODO these currently do NOT follow the requirement that
// after processing probs.sort() == before processing probs.sort()

static void 
add_adjacent_z(ProjMatrixElemsForOneDensel& probs)
{
  // KT&SM 15/05/2000 bug fix !
  // first reserve enough memory for the whole vector
  // otherwise the iterators can be invalidated by memory allocation
  probs.reserve(probs.size() * 3);
  
  ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
  ProjMatrixElemsForOneDensel::const_iterator element_end = probs.end();
  
  while (element_ptr != element_end)
  {      
    probs.push_back( 
      ProjMatrixElemsForOneDensel::value_type(
        Coordinate3D<int>(element_ptr->coord1()+1,element_ptr->coord2(),element_ptr->coord3()),element_ptr->get_value()/2));		 
    probs.push_back( 
      ProjMatrixElemsForOneDensel::value_type(
        Coordinate3D<int>(element_ptr->coord1()-1,element_ptr->coord2(),element_ptr->coord3()),element_ptr->get_value()/2));
	   	   
    ++element_ptr;
  }
}


static void merge_zplus1(ProjMatrixElemsForOneDensel& probs)
{
  // first reserve enough memory to keep everything. 
  // Otherwise iterators might be invalidated.
  probs.reserve(probs.size()*2);
 
  float next_value;
  float current_value = probs.begin()->get_value();
  ProjMatrixElemsForOneDensel::const_iterator lor_old_end = probs.end();
  for (ProjMatrixElemsForOneDensel::iterator lor_iter = probs.begin();
       lor_iter != lor_old_end; 
       ++lor_iter, current_value = next_value)
  {
    // save value before we potentially modify it below
    next_value = (lor_iter+1 == lor_old_end) ? 0.F : (lor_iter+1)->get_value();
    // check if we are the end, or the coordinates of the next voxel are
    // (x,y,z+1)
    if ((lor_iter+1 == lor_old_end) ||
      (lor_iter->coord3() != (lor_iter+1)->coord3()) || 
      (lor_iter->coord2() != (lor_iter+1)->coord2()) ||
      (lor_iter->coord1() + 1 != (lor_iter+1)->coord1()))
    {
      // if not, we can just push_back a new voxel
      probs.push_back(
         ProjMatrixElemsForOneDensel::value_type(
           Coordinate3D<int>(lor_iter->coord1()+1, lor_iter->coord2(), lor_iter->coord3()), 
           current_value));
    }
    else
    {
      // increment value of next voxel with the current value
      *(lor_iter+1) += current_value;
    }
    
  }
  
}

#endif
END_NAMESPACE_STIR

