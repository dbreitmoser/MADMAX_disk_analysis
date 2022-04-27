# MADMAX Booster Disks Measurements

This is a project, dedicated to archive and analyse data gathered by flatness
measurements for the MADMAX disk. 

Measurements come in .txt files with the following information:

Column 1
--------
Run number

Column 2
--------
Hexagon number: 1 - 19

Column 3
--------
Measurement point nr:

* 1..60 within one hexagon
* 13 measurements per point
* first one starts in the middle
* other 12 trace a circle of 1mm diameter around the middle

Column 4
--------
X-coordinate

Column 5
--------
Y-coordinate

Column 6
--------
Z-coordinate

dependencies: 
openpyxl, numpy, pandas, seaborn, matplotlib