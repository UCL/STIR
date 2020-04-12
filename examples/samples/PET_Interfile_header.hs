!INTERFILE  :=
!version of keys := STIR4.0
!imaging modality := PET

name of data file := PET_Interfile_header.s
originating system := unknown
!GENERAL DATA :=
!GENERAL IMAGE DATA :=
; optional keywords specifying patient position
patient rotation := supine
patient orientation := head_in

!type of data := PET
imagedata byte order := LITTLEENDIAN
!PET STUDY (General) :=
!PET data type := Emission
applied corrections := {None}
!number format := float
!number of bytes per pixel := 4
number of dimensions := 4
matrix axis label [4] := segment
!matrix size [4] := 1
matrix axis label [3] := view
!matrix size [3] := 32
matrix axis label [2] := axial coordinate
!matrix size [2] := { 8}
matrix axis label [1] := tangential coordinate
!matrix size [1] := 35
minimum ring difference per segment := { 0}
maximum ring difference per segment := { 0}

; optional keywords specifying frame duration etc
; These are not according to the Interfile 3.3 specification
; Currently only useful in STIR for dynamic applications
; (but a "time frame" is considered to be all projections acquired at the same time)
number of time frames := 1
image duration (sec)[1] := 3
image relative start time (sec)[1] := 1

number of energy windows:=1
energy window lower level[1]:=425
energy window upper level[1]:=650

Scanner parameters:= 
Scanner type := unknown
Energy resolution := 0.145
Reference energy (in keV) := 511

Number of rings                          := 8
Number of detectors per ring             := 64
Inner ring diameter (cm)                 := 65.6
Average depth of interaction (cm)        := 0.7
Distance between rings (cm)              := 3.25
View offset (degrees)                    := 0
end scanner parameters:=
!END OF INTERFILE :=
