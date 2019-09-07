'''
Functions describing the discrete 2 and 3 state network in pair level closure approximation
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import isclose, fsum


def h2(X, Y):
    # '==' can safely be used since we require h2(X,X) = 2 and h(X,Y) = 1 by definition in all cases
    # This also prevents the numerical solution collapsing to zero.
    return 1 + (X == Y)


def h3(X, Y, Z):
    return 1 + (X == Z)


def h4(X, Y, Z, W):
    return 1 + (X == Z) + (X == W) + (Z == W) + (X == Z)*(Z == W) + (X == W)*(Z == W)


print(h4('x', 'x', 'y', 'x'))


def SecondOrderClosure(X, Y, Z, XY, YZ):
    # Moment closure approximation for second order moments
    XYZ = h2(X, Y)*h2(Y, Z)/h3(X, Y, Z) * XY*YZ/Y
    return XYZ


def ThirdOrderClosure(X, Y, Z, W, XY, YZ, YW):
    # Moment closure approximation for third order moments
    XYZW = h2(X, Y)*h2(Y, Z)*h2(Y, W)/h4(X, Y, Z, W) * XY*YZ*YW/(Y**2)
    return XYZW

# Idea was to have Dynamics create the system of ODEs for any M. Was harder than expected, so therefore Dynamics2/3 were created


def Dynamics(moments, t, w0=.2, w2=.2, a=0.5, d=0.1):
    # Input M+1 x M array of [M densities],[ array of M XX, YY pairs] [and array of M(M-1)/2 XY, YZ type of pairs]

    # Unpack list
    stateDensities = moments[0]         # x y
    momentsFirstState = moments[1]      # xx xy
    momentsSecondState = moments[2]     # yy

    M = len(stateDensities)
    momentsDerivative = np.zeros((M+1, M))

    # Create zeroth order moment ODE's
    for index, density in enumerate(stateDensities):
        otherDensities = stateDensities.tolist()
        del otherDensities[index]

        noise = w0 / (M-1) * sum(otherDensities - (M-1)*density)

        stateDynamics = 0
        for otherIndex, otherDensity in enumerate(stateDensities):
            if index != otherIndex:
                stateDynamics += w2 * (SecondOrderClosure(density, otherDensity, density, firstMoments[index, otherIndex], firstMoments[index, otherIndex]) - SecondOrderClosure(
                    otherDensity, density, otherDensity, firstMoments[index, otherIndex], firstMoments[index, otherIndex]))

        momentsDerivative[1, index] = noise + stateDynamics
    return momentsDerivative.flatten()


# def Dynamics2(t, moments, w0, w2, a, d, nLinks=0):
def Dynamics2(moments, t, w0, w2, a, d, nLinks=0):
    X = moments[0]
    Y = moments[1]
    XX = moments[2]
    XY = moments[3]
    YY = moments[4]

    # Higher moments are closed by moment closure approximations
    XYX = SecondOrderClosure(X, Y, X, XY, XY)
    YXY = SecondOrderClosure(Y, X, Y, XY, XY)

    XYXX = ThirdOrderClosure(X, Y, X, X, XY, XY, XY)
    XXYY = ThirdOrderClosure(X, X, Y, Y, XX, XY, XY)
    YXYY = ThirdOrderClosure(Y, X, Y, Y, XY, XY, XY)
    YYXX = ThirdOrderClosure(Y, Y, X, X, YY, XY, XY)

    # ODE's according to Chen
    dXdt = w0 * (Y-X) + w2 * (XYX-YXY)
    dYdt = w0 * (X-Y) + w2 * (YXY-XYX)
    dXXdt = w0 * (XY-2*XX) + w2 * (2*XYX + 3*XYXX-XXYY)
    dYYdt = w0 * (XY-2*YY) + w2 * (2*YXY + 3*YXYY-YYXX)
    dXYdt = w0 * (2*XX+2*YY - 2*XY) + w2 * (-2*XYX - 2*YXY +
                                            XXYY + YYXX - 3*YXYY - 3*XYXX) + a*X*Y - d*XY
    # Returning all derivatives for odeint
    return [dXdt, dYdt, dXXdt, dXYdt, dYYdt]


def Dynamics3(moments, t, w0, w2, a, d):
    M = 3

    # moments = [X, Y, Z, XX, XY, YY, YZ, ZZ, ZX]
    X = moments[0]
    Y = moments[1]
    Z = moments[2]
    XX = moments[3]
    XY = moments[4]
    YY = moments[5]
    YZ = moments[6]
    ZZ = moments[7]
    ZX = moments[8]

    # Higher moments are closed by moment closure approximations
    XYX = SecondOrderClosure(X, Y, X, XY, XY)
    YXY = SecondOrderClosure(Y, X, Y, XY, XY)
    XZX = SecondOrderClosure(X, Z, X, ZX, ZX)
    ZXZ = SecondOrderClosure(Z, X, Z, ZX, ZX)
    YZY = SecondOrderClosure(Y, Z, Y, YZ, YZ)
    ZYZ = SecondOrderClosure(Z, Y, Z, YZ, YZ)

    XYXX = ThirdOrderClosure(X, Y, X, X, XY, XY, XY)
    XXYY = ThirdOrderClosure(X, X, Y, Y, XX, XY, XY)
    YXYY = ThirdOrderClosure(Y, X, Y, Y, XY, XY, XY)
    YYXX = ThirdOrderClosure(Y, Y, X, X, YY, XY, XY)
    XZXX = ThirdOrderClosure(X, Z, X, X, ZX, ZX, ZX)
    XXZZ = ThirdOrderClosure(X, X, Z, Z, XX, ZX, ZX)
    YZYY = ThirdOrderClosure(Y, Z, Y, Y, YZ, YZ, YZ)
    YYZZ = ThirdOrderClosure(Y, Y, Z, Z, YY, YZ, YZ)
    ZXZZ = ThirdOrderClosure(Z, X, Z, Z, ZX, ZX, ZX)
    ZYZZ = ThirdOrderClosure(Z, Y, Z, Z, YZ, YZ, YZ)
    ZZXX = ThirdOrderClosure(Z, Z, X, X, ZZ, ZX, ZX)
    ZZYY = ThirdOrderClosure(Z, Z, Y, Y, ZZ, YZ, YZ)
    YZXX = ThirdOrderClosure(Y, Z, X, X, YZ, ZX, ZX)
    ZXYY = ThirdOrderClosure(Z, X, Y, Y, ZX, XY, XY)
    ZYXX = ThirdOrderClosure(Z, Y, X, X, YZ, XY, XY)
    XZYY = ThirdOrderClosure(X, Z, Y, Y, ZX, YZ, YZ)
    YXZZ = ThirdOrderClosure(Y, X, Z, Z, XY, ZX, ZX)
    XYZZ = ThirdOrderClosure(X, Y, Z, Z, XY, YZ, YZ)

    # ODE's according to Chen
    dXdt = w0/(M-1) * (Y + Z - (M-1)*X) + w2 * ((XYX - YXY) + (XZX - ZXZ))
    dYdt = w0/(M-1) * (Z + X - (M-1)*Y) + w2 * ((YZY - ZYZ) + (YXY - XYX))
    dZdt = w0/(M-1) * (X + Y - (M-1)*Z) + w2 * ((ZXZ - XZX) + (ZYZ - YZY))

    dXXdt = w0/(M-1) * (XY + ZX - 2*(M-1)*XX) + w2 * \
        ((2*XYX + 3*XYXX - XXYY) + (2*XZX + 3*XZXX - XXZZ))
    dYYdt = w0/(M-1) * (XY + YZ - 2*(M-1)*YY) + w2 * \
        ((2*YZY + 3*YZYY - YYZZ) + (2*YXY + 3*YXYY - YYXX))
    dZZdt = w0/(M-1) * (ZX + YZ - 2*(M-1)*ZZ) + w2 * \
        ((2*ZXZ + 3*ZXZZ - ZZXX) + (2*ZYZ + 3*ZYZZ - ZZYY))

    dXYdt = w0/(M-1) * (2*(XX + YY) + ZX + YZ - 2*(M-1)*XY) + w2 * (-2*XYX - 2*YXY +
                                                                    XXYY + YYXX - 3*YXYY - 3*XYXX + (YZXX + XZYY - XYZZ - YXZZ)) + a*X*Y - d*XY
    dYZdt = w0/(M-1) * (2*(YY + ZZ) + ZX + XY - 2*(M-1)*YZ) + w2 * (-2*YZY - 2*ZYZ +
                                                                    YYZZ + ZZYY - 3*ZYZZ - 3*YZYY + (ZXYY + YXZZ - YZXX - ZYXX)) + a*Y*Z - d*YZ
    dZXdt = w0/(M-1) * (2*(ZZ + XX) + XY + YZ - 2*(M-1)*ZX) + w2 * (-2*XZX - 2*ZXZ +
                                                                    XXZZ + ZZXX - 3*ZXZZ - 3*XZXX + (ZYXX + XYZZ - XZYY - ZXYY)) + a*X*Z - d*ZX
    # Returning all derivatives for odeint
    return [dXdt, dYdt, dZdt, dXXdt, dXYdt, dYYdt, dYZdt, dZZdt, dZXdt]
