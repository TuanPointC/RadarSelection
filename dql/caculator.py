import math
import numpy as np


def distance(position_radar, position_target):
    return math.sqrt((position_target.x-position_radar.x)**2+(position_target.y-position_radar.y)**2)

# alpha m, n


def loss_efficients(position_tr, position_rr, position_target):
    distance_tr = distance(position_tr, position_target)
    distance_rr = distance(position_rr, position_target)
    return 1/(distance_rr*distance_tr)**2


def e_m(bandwidth):
    c = 3*10**8
    noise_2 = -110
    return (8*math.pi**2*bandwidth**2)/(noise_2*c**2)

# hệ số có công thức a_m,n * p_m * e_m * |h_m,n|^2


def coefficient(position_tr, position_rr, position_target, power, bandwidth, h):
    return loss_efficients(position_tr, position_rr, position_target)*power*e_m(bandwidth)*h**2


# phân số cho biến x có công thức : (x_radar - x_target)/d_Tm
def fractions_x(position_radar, position_target):
    return (position_radar.x - position_target.x)/distance(position_radar, position_target)

# phân số cho biến y có công thức : (x_radar - x_target)/d_Tm


def fractions_y(position_target, position_radar):
    return (position_radar.y - position_target.y)/distance(position_radar, position_target)

# Công thức (4)


def o_mn(position_tr, position_rr, position_target, power, bandwidth, h):
    return coefficient(position_tr, position_rr, position_target, power, bandwidth, h)*(fractions_x(position_tr, position_target) + fractions_x(position_rr, position_target))**2


# Công thức (5)
def q_mn(position_tr, position_rr, position_target, power, bandwidth, h):
    return coefficient(position_tr, position_rr, position_target, power, bandwidth, h)*(fractions_y(position_tr, position_target) + fractions_y(position_rr, position_target))**2

# Công thức (6)


def r_mn(position_tr, position_rr, position_target, power, bandwidth, h):
    ps_x = fractions_x(position_tr, position_target) + \
        fractions_x(position_rr, position_target)
    ps_y = fractions_y(position_tr, position_target) + \
        fractions_y(position_rr, position_target)
    return coefficient(position_tr, position_rr, position_target, power, bandwidth, h)*ps_x*ps_y

# Ma trận J_mn


def J_mn(position_tr, position_rr, position_target, power, bandwidth, h):
    matrix_J = np.array(
        [o_mn(position_tr, position_rr, position_target, power, bandwidth, h), q_mn(position_tr, position_rr, position_target, power, bandwidth, h), q_mn(position_tr, position_rr, position_target, power, bandwidth, h), r_mn(position_tr, position_rr, position_target, power, bandwidth, h)]).reshape(2, 2)
    return matrix_J

#q_value =trace C(a,b)
def Q_value(list_radar_TRs, list_radar_RRs, position_target, power, bandwidth, h):
    Tr_C_ab = np.array([0, 0, 0, 0]).reshape(2, 2)
    for i in list_radar_TRs:
        for j in list_radar_RRs:
            J_mn = J_mn(i.position, j.position,
                        position_target, power, bandwidth, h)
            Tr_C_ab += i.a*j.a*J_mn

    return Tr_C_ab.trace()
