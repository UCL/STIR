//
// $Id$: $Date$
//

// convecat6_if - conversion from ecat 6 cti to interfile (image and sinogram data)

#include <iostream>
#include <fstream>

#include "InterfileHeader.h"
#include "interfile.h"
#include "PETScannerInfo.h" 
#include "CTI/camera.h"
#include "CTI/cti_types.h"
#include "CTI/cti_utils.h"

// for both image or sinogram data
void write_if_header(char *v_out_name, Main_header v_mhead, FILE *cti_fptr);
// convert image data
void ecat6cti_to_PIOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Image_subheader v_ihead, CameraType camera);
// convert sinogram data
void ecat6cti_to_PSOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Scan_subheader v_shead, CameraType camera);

main(int argc, char *argv[])
{
    char cti_name[50], out_name[50], data_name[50]="";
    FILE *cti_fptr;
 
    if(argc==2) sprintf(cti_name,"%s",argv[1]);
    else {
        cerr<<endl<<"Conversion from ECAT6 CTI data to interfile."<<endl;
        cerr<<"Usage: convecat6_if <data_file_name> (*.img / *.scn / *.*cti)"<<endl<<endl;
        system("ls *.scn *.img *.s2cti *.v2cti");
        cerr<<endl<<"Name of the input data file? : ";
        cin>>cti_name;
    }
    cerr<<endl<<"Name of the output file? (.hv/.hs and .v/.s will be added): ";
    cin>>out_name;

// open input file, read main header
    cti_fptr=fopen(cti_name, "rb"); 
    if(!cti_fptr) {
        cerr<<endl<<"Error opening input file: "<<cti_name<<endl;
        exit(EXIT_FAILURE);
    }
    Main_header mhead;
    if(cti_read_main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
        cerr<<endl<<"Unable to read main header in file: "<<cti_name<<endl;
        exit(EXIT_FAILURE);
    }

    write_if_header(out_name, mhead, cti_fptr);
    fclose(cti_fptr);
    // KT 12/03/2000 added main return value
    return EXIT_SUCCESS;
}


void write_if_header(char *v_out_name, Main_header v_mhead, FILE *cti_fptr)
{
    FILE *hdr_fptr;
    char header_name[50]="",data_name[50]="", *syst_name;
    PETScannerInfo scanner;
    CameraType camera;
    MatDir entry;

    camera=camRPT; // defaulting to camRPT only to read sub headers

    long matnum=cti_numcod(camera,v_mhead.num_frames,v_mhead.num_planes,v_mhead.num_gates,0,v_mhead.num_bed_pos); // get matnum
    if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
        cerr<<endl<<"Couldn't find matnum "<<matnum<<" in specified file."<<endl;
        exit (EXIT_FAILURE);
    }
    
    switch(v_mhead.file_type)
    { 
// IMAGE DATA
        case matImageFile:
          {
            // KT 12/03/2000 moved locally
            Image_subheader ihead;

            if(cti_read_image_subheader(cti_fptr, entry.strtblk, &ihead)!=EXIT_SUCCESS) { // get ihead
                cerr<<endl<<"Unable to look up image subheader"<<endl;
                exit (EXIT_FAILURE);
            }

            // determine camera with dimension parameters (ihead), better than using mhead
            // TODO does not work !
            if(ihead.dimension_1<=129) {
                camera=camRPT;
                scanner=PETScannerInfo::RPT;
            }
            else if(ihead.dimension_1<=161) {
                camera=cam953;
                scanner=PETScannerInfo::E953;
            }
            else {
                camera=cam951;
                scanner=PETScannerInfo::E951;
            }

            sprintf(header_name, "%s.hv", v_out_name); 
            sprintf(data_name, "%s.v", v_out_name);

            // write interfile main header parameters     
            hdr_fptr = fopen(header_name, "w");
            if (!hdr_fptr) {
                cerr<<endl<<"Error opening output header file; "<<header_name<<endl;
                exit(EXIT_FAILURE);
            }

            fprintf(hdr_fptr, "!INTERFILE  :=\n");
            fprintf(hdr_fptr, "name of data file := %s\n", data_name);
            fprintf(hdr_fptr, "!GENERAL DATA :=\n");
            fprintf(hdr_fptr, "!GENERAL IMAGE DATA :=\n");
            fprintf(hdr_fptr, "!type of data := PET\n");
            fprintf(hdr_fptr, "imagedata byte order := %s\n",
              ByteOrder::get_native_order() == ByteOrder::little_endian 
              ? "LITTLEENDIAN"
              : "BIGENDIAN");
            fprintf(hdr_fptr, "!PET STUDY (General) :=\n");
            fprintf(hdr_fptr, "!PET data type := Image\n");
            fprintf(hdr_fptr, "process status :=\n");
            fprintf(hdr_fptr, "!number format := float\n");
            fprintf(hdr_fptr, "!number of bytes per pixel := 4\n");
            fprintf(hdr_fptr, "number of dimensions := 3\n");

            fprintf(hdr_fptr, "!matrix size [1] := %d\n", ihead.dimension_1);
            fprintf(hdr_fptr, "matrix axis label [1] := x\n");
            fprintf(hdr_fptr, "scaling factor (mm/pixel) [1] := %f\n", ihead.pixel_size);
            fprintf(hdr_fptr, "!matrix size [2] :=  %d\n", ihead.dimension_2);
            fprintf(hdr_fptr, "matrix axis label [2] := y\n");
            fprintf(hdr_fptr, "scaling factor (mm/pixel) [2] := %f\n", ihead.pixel_size);
            fprintf(hdr_fptr, "!matrix size [3] :=  %d\n", v_mhead.num_planes);
            fprintf(hdr_fptr, "matrix axis label [3] := z\n");
            fprintf(hdr_fptr, "scaling factor (mm/pixel) [3] := %f\n", scanner.ring_spacing/2);

            fprintf(hdr_fptr, "number of time frames := %d\n", v_mhead.num_frames);
            fprintf(hdr_fptr, "image scaling factor[1] := %f\n", ihead.recon_scale);
            fprintf(hdr_fptr, "quantification units := 1\n");
            fprintf(hdr_fptr, "maximum pixel count := 1\n");
            fprintf(hdr_fptr, "!END OF INTERFILE :=\n");
    
            fclose(hdr_fptr);
            ecat6cti_to_PIOV(data_name, cti_fptr, v_mhead, ihead, camera);
            break;
          }
// SINOGRAM DATA
        case matScanFile:
          {
            // KT 12/03/2000 moved locally, substituted ihead with shead below
            Scan_subheader shead;

            if(cti_read_scan_subheader(cti_fptr, entry.strtblk, &shead)!=EXIT_SUCCESS) { // get shead
                cerr<<endl<<"Unable to look up image subheader"<<endl;
                exit (EXIT_FAILURE);
            }

            // determine camera with dimension parameters (shead), better than using mhead
            if(shead.dimension_1<=128) {
                camera=camRPT;
                scanner=PETScannerInfo::RPT;
                syst_name="PRT-1";
            }
            else if(shead.dimension_1<=160) {
                camera=cam953;
                scanner=PETScannerInfo::E953;
                syst_name="ECAT 953";
            }
            else {
                camera=cam951;
                scanner=PETScannerInfo::E951;
                syst_name="ECAT 951";
            }

            sprintf(header_name, "%s.hs", v_out_name);
            sprintf(data_name, "%s.s", v_out_name);
            
            // write interfile main header parameters
            hdr_fptr = fopen(header_name, "w");
            if (!hdr_fptr) {
                cerr<<endl<<"Error opening output file; "<<header_name<<endl;
                exit(EXIT_FAILURE);
            }

            fprintf(hdr_fptr, "!INTERFILE  :=\n");
            fprintf(hdr_fptr, "name of data file := %s\n", data_name);
            fprintf(hdr_fptr, "originating system := %s\n", syst_name);
            fprintf(hdr_fptr, "!GENERAL DATA :=\n");
            fprintf(hdr_fptr, "!GENERAL IMAGE DATA :=\n");
            fprintf(hdr_fptr, "!type of data := PET\n");
            fprintf(hdr_fptr, "imagedata byte order := %s\n", 
              ByteOrder::get_native_order() == ByteOrder::little_endian 
              ? "LITTLEENDIAN"
              : "BIGENDIAN");
            fprintf(hdr_fptr, "!PET STUDY (General) :=\n");
            fprintf(hdr_fptr, "!PET data type := Emission\n");
            fprintf(hdr_fptr, "!number format := float\n");
            fprintf(hdr_fptr, "!number of bytes per pixel := 4\n");
            fprintf(hdr_fptr, "number of dimensions := 4\n");

            fprintf(hdr_fptr, "!matrix size [1] := 31\n");
            fprintf(hdr_fptr, "matrix axis label [1] := segment\n");
            fprintf(hdr_fptr, "!matrix size [2] := { 16");
            for(int i=15; i>0; i--) fprintf(hdr_fptr, ", %d, %d", i, i);
            fprintf(hdr_fptr, "}\n");
            fprintf(hdr_fptr, "matrix axis label [2] := z\n");
            fprintf(hdr_fptr, "!matrix size [3] :=  %d\n", shead.dimension_2);
            fprintf(hdr_fptr, "matrix axis label [3] := view\n");
            fprintf(hdr_fptr, "!matrix size [4] := %d\n", shead.dimension_1);
            fprintf(hdr_fptr, "matrix axis label [4] := bin\n");

            fprintf(hdr_fptr, "minimum ring difference per segment := {0");
            for(int i=1; i<16; i++) fprintf(hdr_fptr, ", %d, %d", i, -i);
            fprintf(hdr_fptr, "}\n");
            fprintf(hdr_fptr, "maximum ring difference per segment := {0");
            for(int i=1; i<16; i++) fprintf(hdr_fptr, ", %d, %d", i, -i);
            fprintf(hdr_fptr, "}\n");
  
            fprintf(hdr_fptr, "number of rings := %d\n", scanner.num_rings);
            fprintf(hdr_fptr, "number of detectors per ring := %d\n", scanner.num_views*2);
            fprintf(hdr_fptr, "ring diameter (cm) := %f\n", scanner.ring_radius*2/10.);
            fprintf(hdr_fptr, "distance between rings (cm) := %f\n", scanner.ring_spacing/10.);
            fprintf(hdr_fptr, "bin size (cm) := %f\n", scanner.bin_size/10.);
            fprintf(hdr_fptr, "view offset (degrees) := %f\n", scanner.intrinsic_tilt);

            fprintf(hdr_fptr, "number of time frames := %d\n", v_mhead.num_frames);
            fprintf(hdr_fptr, "!END OF INTERFILE :=\n");

            fclose(hdr_fptr);
            ecat6cti_to_PSOV(data_name, cti_fptr, v_mhead, shead, camera);
            break;
          }
        default:
          {
            cerr<<endl<<"Unable to determine file type (either image or scan)."<<endl;
            exit(EXIT_FAILURE);
            break;
          }
    }
}

void ecat6cti_to_PIOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Image_subheader v_ihead, CameraType camera)
{
    FILE *data_file;
    MatDir entry;

    data_file= fopen(v_data_name, "wb");
    int dim_x= v_ihead.dimension_1;
    int dim_y= v_ihead.dimension_2;
    
    // allocation
    short *cti_data= (short *) calloc(dim_x*dim_y, sizeof(short));
    float *if_data= (float *) calloc(dim_x*dim_y, sizeof(float));
    
    for(int z=0; z<v_mhead.num_planes; z++) { // loop upon planes
        long matnum = cti_numcod(camera, v_mhead.num_frames, z+1,v_mhead.num_gates,0,v_mhead.num_bed_pos);
        
        if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
            cerr<<endl<<"Couldn't find matnum "<<matnum<<" in specified file."<<endl;
            exit (EXIT_FAILURE);
        }
        
        if(cti_read_image_subheader(cti_fptr, entry.strtblk, &v_ihead)!=EXIT_SUCCESS) { // get ihead for plane z
            cerr<<endl<<"Unable to look up image subheader"<<endl;
            exit (EXIT_FAILURE);
        }

        if(cti_rblk (cti_fptr, entry.strtblk+1, cti_data, entry.endblk-entry.strtblk)!=EXIT_SUCCESS) { // get data
            cerr<<endl<<"Unable to read data"<<endl;
            exit (EXIT_FAILURE);
        }
    
        swab((char *)cti_data, (char *)cti_data, (entry.endblk-entry.strtblk)*MatBLKSIZE); // swab the bytes
        
        for(int num_pixels=0; num_pixels<dim_y*dim_x; num_pixels++) if_data[num_pixels]= v_ihead.quant_scale*(float)cti_data[num_pixels];
        fwrite(if_data, sizeof(float), dim_y*dim_x, data_file);// write data
    } // end loop upon planes
    free(cti_data);
    fclose(data_file);
}

void ecat6cti_to_PSOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Scan_subheader v_shead, CameraType camera)
{
    FILE *data_file;
    ScanInfoRec scanParams;

    data_file= fopen(v_data_name, "wb");
    int dim_bin= v_shead.dimension_1;
    int dim_view= v_shead.dimension_2;

    // allocation
    short *cti_data= (short *) calloc(dim_bin*dim_view, sizeof(short));
    float *if_data= (float *) calloc(dim_bin*dim_view, sizeof(float));

    // segment order: 0,1,-1,2,-2,...,15,-15
    cerr<<endl<<"Processing segment number:";
    for(int w=0; w<16; w++) { // loop on segment number

// positive ring difference
        cerr<<"  "<<w;
        int num_ring= 16-w;
        // KT 13/03/2000 interchanged
        for(int ring1=0; ring1<num_ring; ring1++) { // ring order: 0-0,1-1,..,15-15 then 0-1,1-2,..,14-15
            int ring2=ring1+w; // ring1<=ring2
            int mat_index= cti_rings2plane(16, ring1, ring2);
            long matnum= cti_numcod(camera, v_mhead.num_frames, mat_index,v_mhead.num_gates,0,v_mhead.num_bed_pos);
            get_scanheaders(cti_fptr, matnum, &v_mhead, &v_shead, &scanParams); // get scanParams
            get_scandata(cti_fptr, cti_data, &scanParams); // read data
            for(int num_pixels=0; num_pixels<dim_view*dim_bin; num_pixels++) 
                if_data[num_pixels]= v_shead.scale_factor*(float)cti_data[num_pixels];
            fwrite(if_data, sizeof(float), dim_view*dim_bin, data_file); // write data
        }

// negative ring difference
        if(w>0) {
            cerr<<"  "<<-w;
            for(int ring2=0; ring2<num_ring; ring2++) { // ring order: 0-1,2-1,..,15-14 then 2-0,3-1,..,15-13
                int ring1=ring2+w; // ring1>ring2
                int mat_index= cti_rings2plane(16, ring1, ring2);
                long matnum= cti_numcod(camera, v_mhead.num_frames, mat_index,v_mhead.num_gates,0,v_mhead.num_bed_pos);
                get_scanheaders(cti_fptr, matnum, &v_mhead, &v_shead, &scanParams); // get scanParams
                get_scandata(cti_fptr, cti_data, &scanParams); // read data
                for(int num_pixels=0; num_pixels<dim_view*dim_bin; num_pixels++) 
                    if_data[num_pixels]= v_shead.scale_factor*(float)cti_data[num_pixels];
                fwrite(if_data, sizeof(float), dim_view*dim_bin, data_file);// write data
            }
        }
    } // end of loop on segment number
    cerr<<endl;
    free(cti_data);
    fclose(data_file);
}



