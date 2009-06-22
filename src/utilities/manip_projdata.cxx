//
// $Id$
//

/*!
\file
\ingroup utilities
\brief process sinogram data

\author Matthew Jacobson
\author Sanida Mustafovic and Kris Thielemans (conversion to new design)
\author PARAPET project

$Date$
$Revision$

This utility programme processes (interfile) sinogram data 
(maximum number of segments as input). It can
<ul>
 <li> display by View - by Segment
 <li> do operations between two data
 <li> do operations with a scalar     
 </ul>
*/
/*
    Copyright (C) 2000 PARAPET partners
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

// TODO get rid of 2 copies of the segments ByView and BySinogram
// TODO get rid of pos, neg segments (can now do each one separately)
// MJ doesn't think doing each one separately is a good idea (for display)



#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"

//#include "stir/Scanner.h"
#include "stir/ArrayFunction.h" 
#include "stir/recon_array_functions.h"
#include "stir/display.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/shared_ptr.h"

#include <numeric>
#include <fstream> 
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
#endif



START_NAMESPACE_STIR

// in relation with show_math_menu()
// _menu HAS to be the last option
enum options { _quit, _display_view, _display_sino, _absdiff, _add_sino, _subtract_sino, 
               _mult_sino, _div_sino, _add_scalar, _mult_scalar, _div_scalar, _stats,
               _pos_ind, _trunc_neg, _trim, _zero_ends, /*_pad_ends,*/ _restart, _menu};

//*********************** prototypes

// operations between two datas
void do_math(enum options operation, SegmentByView<float>& sino1,SegmentByView<float> &sino2,
             float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized);

// display, operations with a scalar, others
void do_math(enum options operation, SegmentByView<float>& sino1, SegmentBySinogram<float>& seg_sinogram, float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized,float scalar=0.0);

void make_buffer_header(const char *data_filename,const char *header_filename, 
                        ProjData& input_sino, int limit_segments, 
			NumericType::Type output_type=NumericType::FLOAT);

void show_math_menu();

float pos_indicate(float x);

shared_ptr<ProjData> ask_proj_data(const char *const input_query);
//*********************** functions

void do_math(enum options operation, SegmentByView<float>& sino1,SegmentByView<float> &sino2,
             float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized)
{
  switch(operation) 
    {

    case _absdiff: { //absolute difference
      sino1-=sino2;
      in_place_abs(sino1);
    
      if(!accumulators_initialized) {
	accum_max=sino1.find_max();
	accum_min=sino1.find_min();
	accum_sum=sino1.sum();
	accumulators_initialized=true;
      }
      else {
	if (accum_max<sino1.find_max()) accum_max= sino1.find_max();
	if (accum_min>sino1.find_min()) accum_min= sino1.find_min();
	accum_sum+=sino1.sum();
      }
      break;
    }
        
    case _add_sino: { // sinogram addition
      sino1+=sino2;
      break;
    }


    case _subtract_sino: { // sinogram subtraction
      sino1-=sino2;
      break;
    }

    case _mult_sino: { // image multiplication
      sino1*=sino2;
      break;
    }

    case _div_sino: { // sinogram division
      divide_array(sino1,sino2);
      break;
    }

    
    //MJ 07/14/2000 empty default to suppress warning in gcc 2.95.2
    default:
      { 
	//empty statement
      }
      


    } // end switch
}


void do_math(enum options operation, SegmentByView<float>& sino1, SegmentBySinogram<float>& seg_sinogram, float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized,float scalar)
{
    switch(operation) 
      {

      case _display_view: { //display math buffer by View
	char title[100];
	sprintf(title, "Segment %d", sino1.get_segment_num());
	display(sino1,sino1.find_max(), title);
	if(ask("Display single viewgram?",false)) {
	  int vs=sino1.get_min_view_num();
	  int ve=sino1.get_max_view_num();
	  int view_num=ask_num("Which viewgram?",vs,ve,vs);
	  
	  Viewgram<float> viewgram=sino1.get_viewgram(view_num);
	  display(viewgram);
	}
	break;
      }

      case _display_sino: { //display math buffer by sinogram
	char title[100];
	sprintf(title, "Segment %d", sino1.get_segment_num());
	display(seg_sinogram, seg_sinogram.find_max());
	break;
      }

      case _add_scalar: { //scalar addition
	sino1+=scalar;
	break;
      }

      case _mult_scalar: { //scalar multiplication
	sino1*=scalar;
	break;
      }

      case _div_scalar: { //scalar division
	sino1/=scalar;
	break;
      }

      case _stats: { //global min&max + number of counts
	if(!accumulators_initialized) {
	  accum_max=sino1.find_max();
	  accum_min=sino1.find_min();
	  accum_sum=sino1.sum();
	  accumulators_initialized=true;
	}
	else {
	  if (accum_max<sino1.find_max()) accum_max= sino1.find_max();
	  if (accum_min>sino1.find_min()) accum_min= sino1.find_min();
	  accum_sum+=sino1.sum();
	}
	break;
      }
        
      case _pos_ind:
	{
	  in_place_apply_function(sino1,pos_indicate); //positive indicator
	  break;
	}
                     
      case _trim: 
	{
	  truncate_rim(sino1, (int) scalar); //trim rim
	  break;
	}

      case _trunc_neg: 
	{
	  in_place_apply_function(sino1,neg_trunc);
	  break;
	}

	
	//MJ 07/14/2000 empty default to suppress warning in gcc 2.95.2
      default:
	{ 
	  //empty statement
	}
	
	  
      } //end switch
}

shared_ptr<ProjData> ask_proj_data(const char *const input_query)
{
    char filename[max_filename_length];

    //system("ls *hs");
    ask_filename_with_extension(filename, input_query, ".hs");

    return 
       ProjData::read_from_file(filename);
}

void show_math_menu()
{
  assert(_menu == 17);

  // KT disabled Pad end planes: 16. Pad end planes of segment 0 \n
cerr<<"\n\
MENU:\n\
0. Quit\n\
1. Display viewgrams\n\
2. Display sinograms\n\
3. Subtract projection array\n\
          and take absolute value\n\
4. Add projection array\n\
5. Subtract projection array\n\
6. Multiply projection array\n\
7. Divide projection array\n\
8. Add scalar\n\
9. Multiply scalar\n\
10. Divide scalar \n\
11. Minimum, maximum & total counts \n\
12. Binarise array (1 if >0, 0 otherwise) \n\
13. Truncate negatives \n\
14. Apply tangential truncating window\n\
       (scalar operand = No. of bins to truncate)\n\
15. Apply axial truncating window to segment 0 \n\
       (scalar operand = No. of planes to truncate)\n\
16. Restart\n\
17. Redisplay menu"<<endl;
}

float pos_indicate(float x)
{
    return (x>0.0)?1.0F:0.0F;
}

END_NAMESPACE_STIR

//********************** main



USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{
    bool quit=false,reload=false;

    shared_ptr<ProjData> first_operand =  NULL;
    ProjDataFromStream *output_proj_data=  NULL;
        // Start
    do { //(re)start from here
        bool buffer_opened=false;
        char output_buffer_header[max_filename_length];

        if (first_operand==NULL) 
	  {

	    if (reload)  
	      // changed the ask... returns ponter 
	      first_operand=ask_proj_data("Input sinogram"); //new
	      //ProjDataFromStream(ask_proj_data("Input sinogram"));

	    else // just starting
	      { 
		if(argc<2)
		  {
		    cerr<<endl<<"Usage: manip_projdata <header file name> (*.hs)"<<endl<<endl;
		    first_operand=ask_proj_data("Input sinogram"); 
		  }
		else first_operand= ProjData::read_from_file(argv[1]);
	  
		reload=false;

	      }

	  }
	 

	int limit_segments=ask_num("Maximum absolute segment number to process: ", 0, first_operand->get_max_segment_num(), first_operand->get_max_segment_num() );


        do { //math operations loop
            float accum_max, accum_min, accum_sum;
            show_math_menu();
            enum options operation;

            operation= 
	      static_cast<options>( 
	        ask_num("Choose Operation: ",
	                                     0,static_cast<int>(_menu), static_cast<int>(_menu))
	        );
            if (operation==_menu) continue; //redisplay menu
            if (operation==_restart || operation==_quit) { //restart or quit
#if 1
	       assert(output_proj_data == NULL);
#else
	      // enable this when using the output buffer for reading/writing at the same time	     
	      if (output_proj_data != NULL)
	      {
		delete output_proj_data;
		output_proj_data = NULL;
	      }
#endif
 	      first_operand=NULL;
	      if(operation==_restart) reload=true;
	      if(operation==_quit) quit=true;
	      break;
            }
  
            if (operation!= _display_view && operation!= _display_sino 
                && operation!= _stats &&!buffer_opened) 
	    {
	      //operation result is a sinogram

              char output_buffer_root[max_filename_length];
              char output_buffer_filename[max_filename_length];
              
	      ask_filename_with_extension(output_buffer_root, "Output to which file (without extension)?", "");
	      sprintf(output_buffer_filename, "%s.%s",output_buffer_root , "s");
	      // TODO relies on write_basic_interfile_PDFS_header using .hs extension
	      sprintf(output_buffer_header, "%s.%s",output_buffer_root , "hs");		
	      fstream * new_sino_ptr = new fstream;
	      open_write_binary(*new_sino_ptr, output_buffer_filename);
	      ProjDataInfo * pdi_ptr =
		first_operand->get_proj_data_info_ptr()->clone();		                         
	      pdi_ptr->reduce_segment_range(-limit_segments, limit_segments);
	      output_proj_data = 
		new ProjDataFromStream(pdi_ptr, new_sino_ptr);
	      write_basic_interfile_PDFS_header(output_buffer_filename, *output_proj_data);
	      buffer_opened=true;
            }

            shared_ptr<ProjData> second_operand= NULL;
            float *scalar=NULL;

            if(operation==_absdiff || operation==_add_sino || operation==_subtract_sino || 
               operation==_mult_sino || operation==_div_sino) //requiring 2nd sinogram operand
                second_operand= ask_proj_data("Second sinogram operand" );
	
            if(operation==_add_scalar || operation==_mult_scalar || operation==_div_scalar ||
               operation==_trim || operation==_zero_ends ) { //requiring scalar operand
                bool need_int=false;
                float upper_bound=100000.F,lower_bound=-100000.F,deflt=0.F;
       
                if(operation==_trim) {
                    need_int=true;
                    upper_bound=(float) (first_operand->get_proj_data_info_ptr()->get_num_tangential_poss()/2 +1);
                    lower_bound=deflt=0.0;
                }

                if(operation==_zero_ends) {
                    need_int=true;
                    upper_bound=(float) (first_operand->get_proj_data_info_ptr()->get_num_axial_poss(0)/2+1);
                    lower_bound=deflt=1.0;
                }

                do scalar= new float (ask_num("Scalar Operand: ",(need_int)?(int)lower_bound:lower_bound ,
                                              (need_int)?(int)upper_bound: upper_bound,(need_int)?(int)deflt:deflt));
                while(*scalar==0.0 && operation==_div_scalar );

            }
// first do segment 0
            { 
                SegmentByView<float> seg1=first_operand->get_segment_by_view(0);
                SegmentBySinogram<float> seg_sinogram=first_operand->get_segment_by_sinogram(0);
#if 0            
		// TODO grow statement is wrong
		// also this can't work anymore, as set_segment would complain about incompatible sizes
		if(operation == _pad_ends)
		  {

		    // TODO this is wrong, as other scanners could have merged segment 0 as well
		    // find out from min_ring_difference etc.
		    bool merges_seg0=(first_operand->get_proj_data_info_ptr()->get_scanner_ptr()->type==
		             Scanner::Advance || 
			     seg1.get_proj_data_info_ptr()->get_scanner_ptr()->type==Scanner::HZLR )? true:false;

		    if((merges_seg0 &&
		        seg1.get_num_axial_poss() == 
			2*(first_operand->get_proj_data_info_ptr()->get_scanner_ptr()->num_rings)-3
			) 
			|| // TODO something wrong here says MJ
			(!merges_seg0 
			 && seg1.get_num_axial_poss() == 
			 first_operand->get_proj_data_info_ptr()->get_scanner_ptr()->num_rings))
		      {
			//seg1.grow_height(seg1.get_min_axial_pos_num()-1,seg1.get_max_axial_pos_num()+1);
		          seg1.grow(seg1.get_min_axial_pos_num()-1,seg1.get_max_axial_pos_num()+1);

		      }
		    else
		      {
			cerr<<"Number of rings is consistent. Operation had no effect"<<endl<<endl;
		      }
		  }
#endif
                if(second_operand.use_count() != 0)  {
                    SegmentByView<float> seg2=second_operand->get_segment_by_view(0);
                    do_math(operation,seg1,seg2,accum_max,accum_min,accum_sum,false);
                }

                else if(scalar != NULL) {
                    if(operation==_zero_ends )
                        for(int i=seg1.get_min_view_num();i<=seg1.get_max_view_num();i++)
                            for(int j=0;j<*scalar;j++ ) {
                                seg1[i][seg1.get_min_axial_pos_num()+j].fill(0);
                                seg1[i][seg1.get_max_axial_pos_num()-j].fill(0);

		
                            }
                    else  do_math(operation,seg1,seg_sinogram,accum_max,accum_min,accum_sum,false,*scalar);


                }

                else do_math(operation,seg1,seg_sinogram,accum_max,accum_min,accum_sum,false);




                    //Write sinogram result to file
                if(operation!= _display_view && operation!= _display_sino && operation!= _stats && buffer_opened) 
		  output_proj_data->set_segment(seg1);
            }
//Now do other segments




            if(limit_segments>0)
                for (int segment_num = 1; segment_num <= limit_segments ; segment_num++) {
                    if((operation==_display_view || operation==_display_sino) && ask("Abort display",false)) break;
                    SegmentByView<float>  seg1_pos=first_operand->get_segment_by_view(segment_num);
                    SegmentByView<float>  seg1_neg=first_operand->get_segment_by_view(-segment_num);
                    SegmentBySinogram<float>  seg_sinogram_pos=first_operand->get_segment_by_sinogram(segment_num);
                    SegmentBySinogram<float>  seg_sinogram_neg=first_operand->get_segment_by_sinogram(-segment_num);
                      
                    if(second_operand.use_count() != 0) {
                        SegmentByView<float> seg2_pos=second_operand->get_segment_by_view(segment_num);
                        SegmentByView<float> seg2_neg=second_operand->get_segment_by_view(-segment_num);
                        do_math(operation,seg1_pos,seg2_pos,accum_max,accum_min,accum_sum,true);
                        do_math(operation,seg1_neg,seg2_neg,accum_max,accum_min,accum_sum,true);
                    }
                    else if(scalar != NULL) {
                        do_math(operation,seg1_pos,seg_sinogram_pos,accum_max,accum_min,accum_sum,true,*scalar);
                        do_math(operation,seg1_neg,seg_sinogram_neg,accum_max,accum_min,accum_sum,true,*scalar);
                    }
                    else {
                        do_math(operation,seg1_pos,seg_sinogram_pos,accum_max,accum_min,accum_sum,true);
                        if((operation==_display_view || operation==_display_sino) 
                           && ask("Abort display",false)) break;
                        do_math(operation,seg1_neg,seg_sinogram_neg,accum_max,accum_min,accum_sum,true);
                        if((operation==_display_view || operation==_display_sino)
                           && segment_num<limit_segments && ask("Abort display",false)) break;
                    }

//Write sinogram result to file
                    if(operation!= _display_view && operation!= _display_sino  && operation!= _stats && buffer_opened) {
                        output_proj_data->set_segment(seg1_neg);  
			output_proj_data->set_segment(seg1_pos);  
                    }
                }


//if buffer changed, reinitialize first operand to output of previous math operation
            if(operation!= _display_view && operation!= _display_sino && operation!= _stats && buffer_opened) 
	    {
	      // at the moment, we close the output buffer, and will reopen it later on
	      // this is to avoid conflicts with reading and writing from/to the same file
	      // alternatively, the output_proj_data would use a read/write file, and
	      // we would do first_operand = output_proj_data
              
	      if (output_proj_data != NULL)
	      {
		delete output_proj_data;
		output_proj_data = NULL;
	      }
	      buffer_opened = false;	      
	      
	      first_operand=ProjData::read_from_file(output_buffer_header);
            }

//Get accumulator results and de-allocate
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Maximum= "<<accum_max<<endl;
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Minimum= "<<accum_min<<endl;
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Total counts= "<<accum_sum<<endl;  
            if (scalar != NULL) delete scalar;

        } while(!quit); // end math operations do-while loop
    } while(!quit); // restart do-while loop

    return EXIT_SUCCESS;
} //end main
