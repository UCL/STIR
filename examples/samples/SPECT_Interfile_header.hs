!INTERFILE  :=
; This is a sample minimal header for SPECT tomographic data
; The format is as per the 3.3 Interfile standard (aside from time frame info)

!imaging modality := nucmed

; name of file with binary data
name of data file := somefile.s

!version of keys := 3.3
!GENERAL DATA :=
!GENERAL IMAGE DATA :=
!type of data := Tomographic

; optional keywords specifying patient position (currently ignored)
; patient rotation := prone
; patient orientation := feet_in

imagedata byte order := LITTLEENDIAN

!SPECT STUDY (General) :=
; specify how the data are stored on disk
; here given as "single-precision float" (you could have "unsigned integer" data instead)
!number format := float
!number of bytes per pixel := 4
!number of projections := 120
; total rotation (or coverage) angle (in degrees)
!extent of rotation := 360
process status := acquired
!SPECT STUDY (acquired data):=
; rotation info (e.g. clock-wise or counter-clock wise)
!direction of rotation := CW
start angle := 180

; Orbit definition
orbit := Circular
; radius in mm
Radius := 150
; or
; orbit := Non-circular
; give a list of "radii", one for every position
; Radius := {150, 151, 153, ....}

; pixel sizes in the acquired data, first in "transverse" direction, then in "axial" direction
!matrix size [1] := 128
!scaling factor (mm/pixel) [1] := 3.32
!matrix size [2] := 64
!scaling factor (mm/pixel) [2] := 6.64

; optional keywords specifying frame duration etc
; These are not according to the Interfile 3.3 specification
; Currently only useful in STIR for dynamic applications
; (but a "time frame" is considered to be all projections acquired at the same time)
;number of time frames := 1
;image duration (sec)[1] := 0
;image relative start time (sec)[1] := 0

!END OF INTERFILE :=
