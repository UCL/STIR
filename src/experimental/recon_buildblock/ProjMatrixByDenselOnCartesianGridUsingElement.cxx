//
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByDenselOnCartesianGridUsingElement

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir_experimental/recon_buildblock/ProjMatrixByDenselOnCartesianGridUsingElement.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/Bin.h"
#include <math.h>

START_NAMESPACE_STIR

void
ProjMatrixByDenselOnCartesianGridUsingElement::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{
  proj_data_info_ptr = proj_data_info_ptr_v;

  const DiscretisedDensityOnCartesianGrid<3,float>* image_info_ptr = 
    dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByDenselOnCartesianGridUsingElement initialised with a wrong type of DiscretisedDensity\n");

 
  grid_spacing = image_info_ptr->get_grid_spacing();
  origin = image_info_ptr->get_origin();
  min_z_index = image_info_ptr->get_min_index(); 
  max_z_index = image_info_ptr->get_max_index(); 


};

//////////////////////////////////////
void 
ProjMatrixByDenselOnCartesianGridUsingElement::
calculate_proj_matrix_elems_for_one_densel(
                                        ProjMatrixElemsForOneDensel& probs) const
{
  const Densel densel = probs.get_densel();

  const float xctr = densel[3] * grid_spacing.x() - origin.x();
  const float yctr = densel[2] * grid_spacing.y() - origin.y();
  const float zctr = 
    (densel[1] - (min_z_index + max_z_index)/2.F) * grid_spacing.z() +
    origin.z();

  const CartesianCoordinate3D<float> densel_ctr(zctr, yctr, xctr);

  assert(probs.size() == 0);

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

    int previous_min_tang_pos = proj_data_info_ptr->get_min_tangential_pos_num();
    // The current logic checks if start_tang_pos gives non-zero. If so, decrement it,
    // otherwise, assume that the non-zero bins lie to the right. This is in some
    // case not true: it's even more to the left, i.e. previous_inc_min_tang_pos was 
    // not negative enough.
    // So, currently I set the increment such that start_tang_pos is guaranteed to be
    // less or equal to proj_data_info_ptr->get_min_tangential_pos_num().
    // This guarantees I don't miss anything, but it's slow...
    int previous_inc_min_tang_pos = 
      proj_data_info_ptr->get_min_tangential_pos_num() - proj_data_info_ptr->get_max_tangential_pos_num();
    const float num_planes_per_axial_pos =
      proj_data_info_ptr->get_sampling_in_t(Bin(seg,0,0,0))*
      sqrt(1+square(proj_data_info_ptr->get_tantheta(Bin(seg,0,0,0))))/
      grid_spacing[1];
    const int min_ax_pos = 
      proj_data_info_ptr->get_min_axial_pos_num(seg) +
      static_cast<int>(floor((densel[1]-max_z_index)/num_planes_per_axial_pos - 1));
    const int max_ax_pos = 
      proj_data_info_ptr->get_max_axial_pos_num(seg) +
      static_cast<int>(ceil((densel[1]-min_z_index)/num_planes_per_axial_pos + 1));
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
            get_element(bin, densel_ctr);
          if (LOI > 0)
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
                bin.set_bin_value(LOI); 
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
              get_element(bin, densel_ctr);
            if (LOI > 0)
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
              get_element(bin, densel_ctr);
          if (LOI > 0)
          {
            if (!found_nonzero_tangential) 
            {
              //std::cerr << "\tfirst tang_pos at " << tang_pos 
              //          << '(' << seg << ',' << view << ',' << ax_pos << ')'<< std::endl;
              //XXXprevious_min_tang_pos = tang_pos;
            }
            found_nonzero_tangential = true;
	    bin.set_bin_value(LOI); 
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
      // next assert only possible when every voxel is detected at least once in every seg,view
      // this would only be true if only voxels in a restricted cylindrical FOV would be used.
      // assert(found_nonzero_axial);
    }
  }
  

#if 0           

  // now add on other LORs
  if ( num_lors_per_axial_pos>1)
  {      
    
    assert(num_lors_per_axial_pos==2);
    if (tantheta==0 ) 
    { 
      assert(Z1f == -origin.z()/grid_spacing.z());
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

