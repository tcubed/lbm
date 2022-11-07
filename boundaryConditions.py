# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:36:57 2022
http://phelafel.technion.ac.il/~drorden/project/ZouHe.pdf

@author: Ted
"""

def zhouHePressure(F,fromdir,k,rho0):
    assert rho0!=0, "rho0 cannot be zero"
    if(fromdir=='w'):
        fsum=F[k+(0,)]+F[k+(2,)]+F[k+(4,)]+2*(F[k+(3,)]+F[k+(6,)]+F[k+(7,)])
        ux=1-fsum/rho0
        ru=ux*rho0
        # nearest-neighbors
        F[k+(1,)]=F[k+(3,)]+2./3*ru
        # next-nearest
        F[k+(5,)]=F[k+(7,)]-(F[k+(2,)]-F[k+(4,)])/2+ru/6
        F[k+(8,)]=F[k+(6,)]+(F[k+(2,)]-F[k+(4,)])/2+ru/6
    elif(fromdir=='e'):
        fsum=F[k+(0,)]+F[k+(2,)]+F[k+(4,)]+2*(F[k+(1,)]+F[k+(5,)]+F[k+(8,)])
        ux=-1+fsum/rho0
        ru=ux*rho0
        # nearest-neighbors
        F[k+(3,)]=F[k+(1,)]-2./3*ru
        # next-nearest
        F[k+(7,)]=F[k+(5,)]+(F[k+(2,)]-F[k+(4,)])/2-ru/6.
        F[k+(6,)]=F[k+(8,)]-(F[k+(2,)]-F[k+(4,)])/2-ru/6.
    elif(fromdir=='s'):
        fsum=F[k+(0,)]+F[k+(1,)]+F[k+(3,)]+2*(F[k+(4,)]+F[k+(7,)]+F[k+(8,)])
        uy=1-fsum/rho0
        ru=uy*rho0
        # nearest-neighbors
        F[k+(2,)]=F[k+(4,)]+2./3*ru
        # next-nearest
        F[k+(5,)]=F[k+(7,)]-(F[k+(1,)]-F[k+(3,)])/2+ru/6
        F[k+(6,)]=F[k+(8,)]+(F[k+(1,)]-F[k+(3,)])/2+ru/6
    elif(fromdir=='n'):
        #fsum=F[k+(0,)]+F[k+(1,)]+F[k+(3,)]+2*(F[k+(1,)]+F[k+(5,)]+F[k+(6,)])
        #uy=-1+fsum/rho0
        fsum=F[k+(0,)]+F[k+(1,)]+F[k+(3,)]+2*(F[k+(2,)]+F[k+(5,)]+F[k+(6,)])
        uy=-1+fsum/rho0
        ru=uy*rho0
        # nearest-neighbors
        F[k+(4,)]=F[k+(2,)]-2./3*ru
        # next-nearest
        F[k+(7,)]=F[k+(5,)]+(F[k+(1,)]-F[k+(3,)])/2-ru/6
        F[k+(8,)]=F[k+(6,)]-(F[k+(1,)]-F[k+(3,)])/2-ru/6
    else:
        raise Exception('not supported')
    return F