//
// $Id$ : $Date$
//

/*!
\file

\brief process sinogram data

\author Matthew Jacobson
\author PARAPET project

\date    $Date$
\version $Revision$

This utility programme processes (interfile) sinogram data 
(maximum number of segments as input). It can
 - display by View - by Segment
 - do operations between two data
 - do operations with a scalar     
*/

#include "pet_common.h"

#include <numeric>

#include "imagedata.h"
#include "TensorFunction.h" 
#include "recon_array_functions.h"
#include "display.h"
#include "interfile.h"
#include "utilities.h"

#define ZERO_TOL 0.0000001

START_NAMESPACE_TOMO

// in relation with show_math_menu()
enum options { _quit, _display_view, _display_segm, _absdiff, _add_sino, _subtract_sino, 
               _mult_sino, _div_sino, _add_scalar, _mult_scalar, _div_scalar, _stats,
               _pos_ind, _trunc_neg, _trim, _zero_ends, _restart, _menu};

//*********************** prototypes

// operations between two datas
void do_math(enum options operation, PETSegmentByView& sino1,PETSegmentByView &sino2,
             float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized);

// display, operations with a scalar, others
void do_math(enum options operation, PETSegmentByView& sino1, PETSegmentBySinogram& segment, float &accum_max, 
             float &accum_min, float &accum_sum, bool accumulators_initialized,float scalar=0.0);

void make_buffer_header(const char *data_filename,const char *header_filename, 
                        PETSinogramOfVolume& input_sino, int limit_segments);

void show_math_menu();

float pos_indicate(float x);

PETSinogramOfVolume ask_interfile_PSOV(char *input_query);
//*********************** functions

void do_math(enum options operation, PETSegmentByView& sino1,PETSegmentByView &sino2,
             float &accum_max, float &accum_min, float &accum_sum, bool accumulators_initialized)
{
    switch(operation) {
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
                if (accum_max>sino1.find_max()) accum_max= sino1.find_max();
                if (accum_min<sino1.find_min()) accum_max= sino1.find_min();
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
    } // end switch
}

void do_math(enum options operation, PETSegmentByView& sino1, PETSegmentBySinogram& segment, float &accum_max,
             float &accum_min, float &accum_sum, bool accumulators_initialized,float scalar)
{
    switch(operation) {
        case _display_view: { //display math buffer by View
            display(sino1, sino1.find_max());
            if(ask("Extract viewgram?",false)) {
                int vs=sino1.get_min_view();
                int ve=sino1.get_max_view();
                int view_num=ask_num("Which viewgram?",vs,ve,vs);
       
                PETViewgram viewgram=sino1.get_viewgram(view_num);
                display(viewgram);
            }
            break;
        }

        case _display_segm: { //display math buffer by Segment
            display(segment, segment.find_max());
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
                if (accum_max>sino1.find_max()) accum_max= sino1.find_max();
                if (accum_min<sino1.find_min()) accum_max= sino1.find_min();
                accum_sum+=sino1.sum();
            }
            break;
        }
        
        case _pos_ind: in_place_apply_function(sino1,pos_indicate); //positive indicator
 
        case _trim: truncate_rim(sino1, (int) scalar); //trim rim

        case _trunc_neg: in_place_apply_function(sino1,neg_trunc);
    
    } //end switch
}

PETSinogramOfVolume ask_interfile_PSOV(char *input_query)
{
    char filename[max_filename_length];

    system("ls *hs");
    ask_filename_with_extension(filename, input_query, ".hs");

    return read_interfile_PSOV(filename);
}

void show_math_menu()
{
    cerr<<"\n\
BINMATH MENU:\n\
0. Quit \n\
1. Display by View\n\
2. Display by Segment\n\
3. Absolute difference\n\
4. Add sinogram\n\
5. Subtract sinogram\n\
6. Multiply sinogram\n\
7. Divide sinogram\n\
8. Add scalar\n\
9. Multiply scalar\n\
10. Divide scalar \n\
11. Min/Max & counts \n\
12. Positive indicator \n\
13. Truncate negatives \n\
14. Trim end bins\n\
15. Zero end planes of segment 0 \n\
16. Restart\n\
17. Redisplay menu"<<endl;
}


void make_buffer_header(const char *data_filename,const char *header_filename,
                        PETSinogramOfVolume& input_sino, int limit_segments)
{
    // TODO replace by write_interfile_PSOV_header

    int nrings=input_sino.scan_info.get_num_rings();
    ofstream header;
    header.open(header_filename, ios::out);
    if(!header) {
        cerr<<endl<<"Could not create header"<<endl;
        exit(1);
    }

    header<<"!INTERFILE  := \n";
    header<<"name of data file := "<<data_filename;
    header<<"\n";
    header<<"originating system := ";
//Get the Scanner name
    if(input_sino.scan_info.get_scanner().type==PETScannerInfo::RPT) header<<"PRT-1 \n";
    else if(input_sino.scan_info.get_scanner().type==PETScannerInfo::Advance) header<<"Advance \n";
    else if(input_sino.scan_info.get_scanner().type==PETScannerInfo::E953) header<<"ECAT 953 \n";
    else if(input_sino.scan_info.get_scanner().type==PETScannerInfo::E951) header<<"ECAT 951 \n";
    else if(input_sino.scan_info.get_scanner().type==PETScannerInfo::E966) header<<"EXACT3D \n";
    else { 
        error("Tried to create header for unsupported scanner type");
    }


    header<<"!GENERAL DATA := \n";
    header<<"!GENERAL IMAGE DATA := \n";
    header<<"!type of data := PET \n";
    header<<"imagedata byte order := " <<
      (ByteOrder::get_native_order() == ByteOrder::little_endian 
       ? "LITTLEENDIAN"
       : "BIGENDIAN")<< "\n";
    header<<"!PET STUDY (General) := \n";
    header<<"!PET data type := Emission \n";
    header<<"!number format := float \n";
    header<<"!number of bytes per pixel := 4 \n"; 
    header<<"number of dimensions := 4 \n";
    header<<"!matrix size [1] := " <<2*limit_segments+1<<"\n"; 
    header<<"matrix axis label [1] := segment \n"; 
    header<<"!matrix size [2] := "<<input_sino.scan_info.get_num_views()<<"\n"; 
    header<<"matrix axis label [2] := view \n";
//number of rings per segment
    header<<"!matrix size [3] := { ";
    if(input_sino.scan_info.get_scanner().type==PETScannerInfo::Advance) {
        header<<2*nrings-1;
        for (int i=1; i<=limit_segments; ++i) header<<", "<<nrings-(i+1)<<", "<<nrings-(i+1);
    }
    else {
        header<< nrings;
        for (int i=1; i<=limit_segments; ++i) header<<", "<<nrings-i<<", "<<nrings-i;
    }
    header<<"}\n";
//end -- number of  rings per segment
    header<<"matrix axis label [3] := z \n"; 
    header<<"!matrix size [4] := "<< input_sino.scan_info.get_num_bins()<<"\n";
    header<<"matrix axis label [4] := bin \n";
//min ring differences per segment
    header<<"minimum ring difference per segment := {";
    if(input_sino.scan_info.get_scanner().type==PETScannerInfo::Advance) {
        header<<"-1";
        for (int i=1; i<=limit_segments; ++i) header<<", "<<(i+1)<<", "<<-(i+1);
    }
    else {
        header<<"0";
        for (int i=1; i<=limit_segments; ++i) header<<", "<<i<<", "<<-i;
    }
    header<<"}\n";
//end min ring differences

//max ring differences per segment
    header<<"maximum ring difference per segment := {";
    if(input_sino.scan_info.get_scanner().type==PETScannerInfo::Advance) {
        header<<"1";
        for (int i=1; i<=limit_segments; ++i) header<<", "<<(i+1)<<", "<<-(i+1);
    }
    else {
        header<<"0";
        for (int i=1; i<=limit_segments; ++i) header<<", "<<i<<", "<<-i;
    }
    header<<"}\n";
//end max ring differences
    header<<"number of rings := "<<nrings<<"\n"; 
    header<<"number of detectors per ring := "<<2*input_sino.scan_info.get_num_views()<<"\n";

    header.setf(ios::fixed);

    header<<"ring diameter (cm) := "<<2*input_sino.scan_info.get_ring_radius()/10.<<"\n";
    header<<"distance between rings (cm) := "<<input_sino.scan_info.get_ring_spacing()/10.<<"\n";
    header<<"bin size (cm) := "<< input_sino.scan_info.get_bin_size()/10.<<"\n";
    header<<"view offset (degrees) := " <<input_sino.scan_info.get_scanner().intrinsic_tilt<<"\n";

    header.unsetf(ios::fixed);

    header<<"number of time frames := 1\n"; 
    header<<"!END OF INTERFILE :="<<"\n";

    header.close();
}


float pos_indicate(float x)
{
    return (x>0.0)?1.0:0.0;
}

END_NAMESPACE_TOMO

//********************** main

USING_NAMESPACE_TOMO

int main(int argc, char *argv[])
{
    bool quit=false,reload=false,abort_display;

    PETSinogramOfVolume *first_operand =  NULL;
        // Start
    do { //(re)start from here
        ofstream new_sino;
        char output_buffer_root[max_filename_length];
        char output_buffer_header[max_filename_length];
        char output_buffer_filename[max_filename_length];
        bool buffer_opened=false;

        if (reload) {
            first_operand= new PETSinogramOfVolume(ask_interfile_PSOV("Input sinogram"));
            reload=false;
        }
        else if(argc>1) first_operand= new PETSinogramOfVolume(read_interfile_PSOV(argv[1]));
        else {
            cerr<<endl<<"Usage: binmath <header file name> (*.hs)"<<endl<<endl;
            first_operand= new PETSinogramOfVolume(ask_interfile_PSOV("Input sinogram"));
        }

        int limit_segments=(argc>2)? atoi(argv[2]):ask_num("Maximum absolute segment number to process: ", 
                                                           0, first_operand->get_max_segment(), first_operand->get_max_segment() );

        do { //math operations loop
            float accum_max, accum_min, accum_sum;
            show_math_menu();
            enum options operation;

            operation= (enum options) ask_num("Choose Operation: ",0,17,17);
            if (operation==_menu) continue; //redisplay menu
            if (operation==_restart || operation==_quit) { //restart or quit
                new_sino.close();
                assert(first_operand != NULL);
                delete first_operand;
                if(operation==_restart) reload=true;
                if(operation==_quit) quit=true;
                break;
            }
  
            if (operation!= _display_view && operation!= _display_segm 
                && operation!= _stats &&!buffer_opened) {  //operation result is a sinogram
                ask_filename_with_extension(output_buffer_root, "Output to which file (without extension)?", "");
                sprintf(output_buffer_filename, "%s.%s",output_buffer_root , "s");
                sprintf(output_buffer_header, "%s.%s",output_buffer_root , "hs");
                make_buffer_header(output_buffer_filename, output_buffer_header, *first_operand, limit_segments); 
                // first_operand->get_max_segment()
                open_write_binary(new_sino, output_buffer_filename);
                buffer_opened=true;
            }

            PETSinogramOfVolume *second_operand= NULL;
            float *scalar=NULL;

            if(operation==_absdiff || operation==_add_sino || operation==_subtract_sino || 
               operation==_mult_sino || operation==_div_sino) //requiring 2nd sinogram operand
                second_operand= new PETSinogramOfVolume(ask_interfile_PSOV("Second sinogram operand" ));
            if(operation==_add_scalar || operation==_mult_scalar || operation==_div_scalar ||
               operation==_trim || operation==_zero_ends ) { //requiring scalar operand
                bool need_int=false;
                float upper_bound=100000.F,lower_bound=-100000.F,deflt=0.F;
       
                if(operation==_trim) {
                    need_int=true;
                    upper_bound=(float) (first_operand->scan_info.get_num_bins()/2 +1);
                    lower_bound=deflt=0.0;
                }

                if(operation==_zero_ends) {
                    need_int=true;
                    upper_bound=(float) (first_operand->scan_info.get_num_rings()/2+1);
                    lower_bound=deflt=1.0;
                }

                do scalar= new float (ask_num("Scalar Operand: ",(need_int)?(int)lower_bound:lower_bound ,
                                              (need_int)?(int)upper_bound: upper_bound,(need_int)?(int)deflt:deflt));
                while(*scalar==0.0 && operation==_div_scalar );

            }
// first do segment 0
            { 
                PETSegmentByView seg1=first_operand->get_segment_view_copy(0);
                PETSegmentBySinogram segment=first_operand->get_segment_sino_copy(0);
                
                    // PETSegmentByView *seg2 =NULL;
                if(second_operand != NULL)  {
                    PETSegmentByView seg2=second_operand->get_segment_view_copy(0);
                    do_math(operation,seg1,seg2,accum_max,accum_min,accum_sum,false);
                }

                else if(scalar != NULL) {
                    if(operation==_zero_ends )
                        for(int i=seg1.get_min_view();i<=seg1.get_max_view();i++)
                            for(int j=0;j<*scalar;j++ ) {
                                seg1[i][seg1.get_min_ring()+j].fill(0);
                                seg1[i][seg1.get_max_ring()-j].fill(0);
                            }
                    else  do_math(operation,seg1,segment,accum_max,accum_min,accum_sum,false,*scalar);
                }
                else do_math(operation,seg1,segment,accum_max,accum_min,accum_sum,false);

                    //Write sinogram result to file
                if(operation!= _display_view && operation!= _display_segm && operation!= _stats && buffer_opened) seg1.write_data(new_sino);
            }
//Now do other segments
            if(limit_segments>0)
                for (int segment_num = 1; segment_num <= limit_segments ; segment_num++) {
                    if((operation==_display_view || operation==_display_segm) && ask("Abort display",false)) break;
                    PETSegmentByView  seg1_pos=first_operand->get_segment_view_copy(segment_num);
                    PETSegmentByView  seg1_neg=first_operand->get_segment_view_copy(-segment_num);
                    PETSegmentBySinogram  segment_pos=first_operand->get_segment_sino_copy(segment_num);
                    PETSegmentBySinogram  segment_neg=first_operand->get_segment_sino_copy(-segment_num);
                      
                    if(second_operand != NULL) {
                        PETSegmentByView seg2_pos=second_operand->get_segment_view_copy(segment_num);
                        PETSegmentByView seg2_neg=second_operand->get_segment_view_copy(-segment_num);
                        do_math(operation,seg1_pos,seg2_pos,accum_max,accum_min,accum_sum,true);
                        do_math(operation,seg1_neg,seg2_neg,accum_max,accum_min,accum_sum,true);
                    }
                    else if(scalar != NULL) {
                        do_math(operation,seg1_pos,segment_pos,accum_max,accum_min,accum_sum,true,*scalar);
                        do_math(operation,seg1_neg,segment_neg,accum_max,accum_min,accum_sum,true,*scalar);
                    }
                    else {
                        do_math(operation,seg1_pos,segment_pos,accum_max,accum_min,accum_sum,true);
                        if((operation==_display_view || operation==_display_segm) 
                           && ask("Abort display",false)) break;
                        do_math(operation,seg1_neg,segment_neg,accum_max,accum_min,accum_sum,true);
                        if((operation==_display_view || operation==_display_segm)
                           && segment_num<limit_segments && ask("Abort display",false)) break;
                    }

//Write sinogram result to file
                    if(operation!= _display_view && operation!= _display_segm  && operation!= _stats && buffer_opened) {
                        seg1_neg.write_data(new_sino);  
                        seg1_pos.write_data(new_sino);
                    }
                }

//if buffer changed, reinitialize put pointer and update first operand
            if(operation!= _display_view && operation!= _display_segm && operation!= _stats && buffer_opened) {
                new_sino.seekp(0,ios::beg);
                assert(first_operand != NULL);
                delete first_operand;
                first_operand=new PETSinogramOfVolume(read_interfile_PSOV(output_buffer_header));
            }

//Get accumulator results and de-allocate
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Maximum= "<<accum_max<<endl;
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Minimum= "<<accum_min<<endl;
            if (operation ==_absdiff || operation ==_stats) cerr<<endl<<"Total counts= "<<accum_sum<<endl;  
            if (second_operand != NULL) delete second_operand;
            if (scalar != NULL) delete scalar;

        } while(!quit); // end math operations do-while loop
    } while(!quit); // restart do-while loop

    assert(first_operand != NULL);
    delete first_operand;
    return EXIT_SUCCESS;
} //end main
