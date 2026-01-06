# FFTix

This repository contains some tools for FFT-based homogenization. The idea is to use numpy.fft and hence providing simple codes with a simple structure. 


utils.py : Utilities for manipulation of tensors (requires numba)

Operators.py : Green's operators and so on (requires numba)

Linear_schemes.py : well-known FFT-based iterative schemes for homogenization in mechanics (and thermics: coming soon).

Non_linear_schemes.py : algorithms for integration of a non linear behaviour (imposed macroscopic strain)

demos : folder with demos
 
 - linear_elasticity.py : simple examples for linear case
 - linear_viscoelasticity.py : example of a viscoelastic computation, using MFront material behaviours on each phase. The file "maxwell.mfront" must be compiled with mfront by doing "mfront --obuild --interface=generic maxwell.mfront" and then, to launch the demo (but also to use Non_linear_schemes.py), the mgis module must be sourced. Hence, you must install MFrontGenericInterfaceSupport.

 ## Examples

 Here is an example with Linear_schemes.py, of a FFT computation in linear elasticity, with "Brisard-Dormieux" discretization of Green operator, on a 256^3 grid:

 <p align="center">
    <img src="images/fibrous.png" width="60%" /><br>
    <em> Reinforced medium, FFT computation</em>
 </p>

 Here is another example with Non_linear_schemes.py, of a FFT computation in linear visco-elasticity:
 
 <p align="center">
    <video src="https://github.com/Antoine2022/FFTix/blob/main/images/visco.mp4" width="60%" /><br>
    <em> Reinforced viscoelastic medium, FFT computation</em>
 </p>

