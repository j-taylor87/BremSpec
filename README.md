# BremSpec 

https://bremspec.streamlit.app/

Interactive Bremsstrahlung X-ray Spectrum Visualiser built using streamlit and attenuation data from NIST XCOM https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html.

## What is the Bremmstrahlung spectrum? 
Bremsstrahlung means "braking radiation" in German, and the spectrum is named so for the X-rays that are produced when electrons strike a target.
The incident, negatively charged electrons interact with the nuclei of the target, causing the electrons to decelerate or "brake". 
This loss in electron kinetic energy is converted into X-rays (conservation of energy), which are then used for imaging.
Read more here: https://en.wikipedia.org/wiki/Bremsstrahlung

Select an X-ray imaging modality, filter materials, and technique factors (kV, mA, s) to see how these affect the shape of the Bremsstrahlung X-ray spectrum.
The characteristic X-rays of the target material and median beam energy can also be displayed. 
The relative area under the curve (AUC), representing the total beam intensity across all available energies, is displayed in the top-right corner of the plot.
