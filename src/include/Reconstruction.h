// $Id$: $Date$
//
#ifndef __RECONSTRUCTION_H__
#define __RECONSTRUCTION_H__

#include "pet_common.h"
#include <string>

#include "sinodata.h"
#include "imagedata.h"
#include "Filter.h"


//	PETSinogramOfVolume s(/* appropriate parameters*/);
//	PETImageVolume v(/* appropriate parameters*/);
//		
//	recon.reconstruct(s,v);


// KT 28/06/98 disabled for now
#if 0
// CL 1/12 These two fonctions should be declared somewherre else
PETSinogramOfVolume CorrectForAttenuation(const PETSinogramOfVolume &sino, const PETSinogramOfVolume &atten)       
{
// Kernel of Attenuation correction
    PETerror("TODO"); Abort();
    return sino;
}

PETSegmentBySinogram CorrectForAttenuation(const PETSegmentBySinogram &sino, const PETSegmentBySinogram &atten)
{
// Kernel of Attenuation correction
    PETerror("TODO"); Abort();
    return sino;
}
#endif

class PETReconstruction
{
 
    
public:
        // KT 02/06/98 made pure virtual
    virtual string method_info();

    virtual string parameter_info()
        { return ""; }
        /* CL 15/10/98 Default constructor*/
    PETReconstruction(){
        cout << "Reconstruction starting..." << endl;
    }
 
  
    virtual void reconstruct(const PETSinogramOfVolume &s, PETImageOfVolume &v)
        {}
  
// KT 28/06/98 disabled for now
#if 0
    virtual void  reconstruct(const PETSinogramOfVolume &s, const PETSinogramOfVolume &a, PETImageOfVolume &v) 
        { 
            reconstruct(CorrectForAttenuation(s, a), v); 
        }
#endif
};

class PETAnalyticReconstruction: public PETReconstruction
{
public:
    int delta_min; /* Lower bound of "obliqueness" */
    int delta_max; /* Upper bound of "obliqueness" */
   const Filter1D<float> &filter; /* Ramp filter */
    
        //CL 15/10/98 New function
    virtual string method_info(){
        return("Analytical algorithm");
    }
  
    virtual string parameter_info();
    
    
    PETAnalyticReconstruction(int min, int max, const Filter1D<float>& f);
    
};

class PETIterativeReconstruction: public PETReconstruction
{
public:
    int max_iterations;
    PETIterativeReconstruction(int max=1);
            //CL 15/10/98 New function
    virtual string method_info();
};

inline PETIterativeReconstruction::PETIterativeReconstruction
(int max) :max_iterations(max)
{    
}



/******************* 2D reconstructions ************/
class PETReconstruction2D
{
public:
        // KT 02/06/98 made pure virtual
    virtual string method_info() = 0;
    virtual string parameter_info()
        { return ""; }
  
    virtual void reconstruct(const PETSinogram &sino2D, PETPlane &image2D)=0;
    virtual void reconstruct(const PETSegment &sino, PETImageOfVolume &image)=0;  
        // KT 28/06/98 disabled for now
#if 0
    virtual void reconstruct(const PETSegment& s, 
                             const PETSegment& a, PETImageOfVolume& v) 
        { reconstruct(CorrectForAttenuation(s, a), v); }
#endif
};





class PETAnalyticReconstruction2D: public PETReconstruction2D
{
protected:
    const Filter1D<float> &filter;
public:
    virtual string parameter_info();
        //CL 15/10/98 ACtivate filter
              PETAnalyticReconstruction2D(const Filter1D<float>& f);    
    
};



class Reconstruct2DFBP : public PETAnalyticReconstruction2D
{

public:
    virtual string method_info()
        { return("2DFBP"); }
 
    Reconstruct2DFBP(const Filter1D<float>& f): PETAnalyticReconstruction2D(f) 
        {
            cout << "  - 2D FBP processing" << endl;
        }
    

    void reconstruct(const PETSinogram &sino2D, PETPlane &image2D);
    void reconstruct(const PETSegment &sino, PETImageOfVolume &image);
};


    
#endif
