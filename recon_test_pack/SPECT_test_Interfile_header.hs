!INTERFILE  :=
; This is a sample minimal header for SPECT tomographic data
; The format is as per the 3.3 Interfile standard (aside from time frame info)

!imaging modality := nucmed

; name of file with binary data
name of data file := SPECT_test_Interfile_header.s

!version of keys := 3.3
!GENERAL DATA :=
!GENERAL IMAGE DATA :=
!type of data := Tomographic

; optional keywords specifying patient position
; patient rotation := prone
; patient orientation := feet_in

imagedata byte order := LITTLEENDIAN

number of energy windows:=1
energy window lower level[1]:=120
energy window upper level[1]:=160

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
Radius := 166.5
; or
; orbit := Non-circular
; give a list of "radii", one for every position
; Radii := {150, 151, 153, ....}

; pixel sizes in the acquired data, first in "transverse" direction, then in "axial" direction
!matrix size [1] := 111
!scaling factor (mm/pixel) [1] := 3
!matrix size [2] := 47
!scaling factor (mm/pixel) [2] := 3.27

; optional keywords specifying frame duration etc
; These are not according to the Interfile 3.3 specification
; Currently only useful in STIR for dynamic applications
; (but a "time frame" is considered to be all projections acquired at the same time)
;number of time frames := 1
;image duration (sec)[1] := 0
;image relative start time (sec)[1] := 0

!END OF INTERFILE :=
