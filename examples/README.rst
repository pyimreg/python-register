.. -*- mode: rst -*-

About
========

Demonstrates uses of the sci-kit:
  
  Examples of 2D linear and nonlinear image registration.
  
  + linreg
  
  Uses an affine deformation model to deform the template image and then (attempts to) minimize:

  || T - W(F;p) ||
  
  where:
     
     T : target image (lena deformed)
     F : floating image ( non-deformed image)
     p : warp parameters
  
  + nonlinreg
  
  Uses an cubic spline deformation model to defrom the template image and then (attempts to) minimize:
  
  || T - W(F;p) || + a*||p||
  
  where:
     
     T : target image (lena deformed)
     F : floating image ( non-deformed image)
     p : warp parameters
     a : regularization term - determines smoothness of the warping.
  

