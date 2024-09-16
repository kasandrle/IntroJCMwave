import numba
import numpy as np


@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def corner_round(x1, x2, x3, r, n=50):
    """
    Calculates the x and y points of a rounded corner
    :param x1: 2d array with x,y of point 1
    :param x2: 2d array with x,y of point 2 (this is the point in the middle)
    :param x3: 2d array with x,y of point 3
    :param r: radius of the corner
    :param n: number of points of the corner
    :return: array containing the x,y of the rounded corner
    """
    #direction vector from x2 to x1 and x2 to x3
    a = np.empty(2)
    b = np.empty(2)

    a[0] = x2[0] - x1[0]
    a[1] = x2[1] - x1[1]
    b[0] = x2[0] - x3[0]
    b[1] = x2[1] - x3[1]

    # norm of a and b
    norme_a = np.linalg.norm(a)
    norme_b = np.linalg.norm(b)
    # normded vector a and b
    norm_a = a / norme_a
    norm_b = b / norme_b
    # angles between a and b
    ang = np.arccos(np.dot(a, b) / norme_a / norme_b)
    # angle of the a vector to the x-axis, needed as an offset
    ang_0 = np.arccos(a[0] / norme_a)
    # angular apertur of the corner
    beta2 = np.pi - ang
    # distance between x2 and the center of the circle of the corner and x2
    c = -r / np.sin(ang / 2)
    # vector pointing along the line between the center of the circle of the corner and x2
    direct = (norm_a + norm_b) / np.linalg.norm(norm_a + norm_b)
    # calculating the x and y of the center of the circle
    c_point = x2 + c * direct
    # look in wich part of the circle the corner need to be
    if direct[0] > 0 and direct[1] > 0:
        rot = -1
    elif direct[0] < 0 and direct[1] < 0:
        rot = 0
    elif direct[0] < 0 and direct[1] > 0:
        rot = -1
    else:
        rot = 2

    result = np.empty((n, 2))
    # angle offset
    corr_ang = rot / 2 * np.pi + ang_0
    # calculate the points of the corner
    for i in range(n):
        result[i][0] = np.cos(beta2 / n * i + corr_ang) * r + c_point[0]
        result[i][1] = np.sin(beta2 / n * i + corr_ang) * r + c_point[1]
    # look if the points are in the right order
    if a[0] * b[1] - a[1] * b[0] > 0:
        result = result[::-1]

    return result

@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def do_layers(points_oxid_x_o,points_oxid_x,points_oxid_y_o,points_oxid_y,thickness_oxid_etch_offset=0,R=5,R1=5,N1 = 2, dN1 = 4,n=10):
    ee_x = points_oxid_x[:N1 + 1]
    ee_y = points_oxid_y[:N1 + 1]
    
    for i in range(N1, N1 + dN1):
        if i + 2 < np.shape(points_oxid_x_o)[0]:
            X1 = np.array([points_oxid_x_o[i], points_oxid_y_o[i]])
            X2 = np.array([points_oxid_x_o[i + 1], points_oxid_y_o[i + 1]])
            X3 = np.array([points_oxid_x_o[i + 2], points_oxid_y_o[i + 2]])
        elif i + 1 < np.shape(points_oxid_x_o)[0]:
            X1 = np.array([points_oxid_x_o[i], points_oxid_y_o[i]])
            X2 = np.array([points_oxid_x_o[i + 1], points_oxid_y_o[i + 1]])
            X3 = np.array([points_oxid_x_o[i + 2 - np.shape(points_oxid_x_o)[0]],
                               points_oxid_y_o[i + 2 - np.shape(points_oxid_x_o)[0]]])
        if (i == N1 or i == N1 + dN1 -1) and N1 > 1:
            eroded = corner_round(X1, X2, X3, R1, n)
        else:
            eroded = corner_round(X1, X2, X3, R, n)
        points_oxid_x = np.delete(points_oxid_x, N1 + 1)
        points_oxid_y = np.delete(points_oxid_y, N1 + 1)
        ee_x = np.concatenate((ee_x.ravel(), (eroded[:, 0]).ravel()))
        ee_y = np.concatenate((ee_y.ravel(), (eroded[:, 1]).ravel()))

    points_oxid_x = np.concatenate((ee_x.ravel(), points_oxid_x[N1 + 1:].ravel()))
    points_oxid_y = np.concatenate((ee_y.ravel(), points_oxid_y[N1 + 1:].ravel()))
        # remove self intersecting points
    while len(points_oxid_x) > 0:
        x_array_diff = np.diff(points_oxid_x)
        if len(np.where(x_array_diff > 0)[0]) > 1:
            rm = np.where(x_array_diff > 0)[0][1]
            points_oxid_x = np.delete(points_oxid_x, [rm, rm + 1])
            points_oxid_y = np.delete(points_oxid_y, [rm, rm + 1])
        else:
            break

    points_oxid_new = np.array([1.1])

        #j = 0
    for i in range(np.shape(points_oxid_x)[0]):
        points_oxid_new = np.concatenate((points_oxid_new.ravel(), np.array([points_oxid_x[i]]).ravel()))
        points_oxid_new = np.concatenate((points_oxid_new.ravel(), np.array([points_oxid_y[i]]).ravel()))
    points_oxid_komplet = points_oxid_new[1:]
  
            
    return points_oxid_komplet

@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def calc_shift(distance, swa_rad):
    dy = np.abs(distance*np.sin(np.pi/2-swa_rad))
    dx_base = np.abs(distance*np.sin(swa_rad))
    dx = np.sqrt(np.square(distance)-np.square(dy))
    return dy, dx, dx_base


@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def geometry(cd, R, thickness_oxid, swa, pitch, height, height_offset_substrate, thickness_oxid_Si_Si3N4,
             height_etch_offset, thickness_oxid_etch_offset, height_offset_air,height_C=0, height_C_etch_offset =0, R1 = -1,n=10):
    """
    Calculates all needed point for the jcm layout
    :param cd: width of the grating
    :param R: radius of the corners
    :param thickness_oxid: thickness of the oxide layer
    :param swa: side wall angle
    :param pitch: pitch of the grating
    :param height: height of the grating
    :param height_offset_substrate:  height of the offset of the grating
    :param thickness_oxid_Si_Si3N4: thickness of the oxide layer of the substrate(Si)
    :param height_etch_offset: height of the material that has not been etched
    :param thickness_oxid_etch_offset: height of the oxide in the grooves
    :param height_offset_air: height of the air/vacuum above the grating
    :return:
    """

    if R1 == -1:
        R1 = R
        
    swa_rad = np.deg2rad(swa)
    dyo, dxo, dxo_base = calc_shift(thickness_oxid,swa_rad)
    dyoc, dxoc, dxoc_base = calc_shift(thickness_oxid+height_C,swa_rad)

    x1 = -0.5 * pitch
    x2 = -0.5 * cd - (height / 2) * np.tan(np.deg2rad(90 - swa))
    x3 = -0.5 * cd + (height / 2) * np.tan(np.deg2rad(90 - swa))
    
    x2o = x2 - dxo_base#x2 + dx
    x3o = x3 - dxo#x3 + dx
    x4o = -x3o
    x5o = -x2o
    
    x2oc = x2 - dxoc_base #x2 + dxc
    x3oc = x3 - dxoc #x3 + dxc
    x4oc = -x3oc
    x5oc = -x2oc

    #print(x2,x2o,x2oc)

    x4 = -x3
    x5 = -x2
    x6 = -x1

    y1 = -height_offset_substrate - thickness_oxid_Si_Si3N4
    y12 = -thickness_oxid_Si_Si3N4
    y2 = 0.
    y3 = 0. + height_etch_offset
    y3o = 0. + height_etch_offset + thickness_oxid_etch_offset
    y3oc = 0. + height_etch_offset + thickness_oxid_etch_offset + height_C_etch_offset
    
    y4 = height
    y4o = height + thickness_oxid
    y4oc = height+ thickness_oxid+ height_C
    y5 = height+ thickness_oxid+ height_C + height_offset_air

    points_cd = np.array([x1, y1, x6, y1, x6, y5, x1, y5])
    points_substrate = np.array([x1, y1, x6, y1, x6, y2, x1, y2])

    if thickness_oxid_Si_Si3N4 > 0:
        points_to_Si_Si3N4 = np.array([x1, y12, x6, y12, x6, y2, x1, y2])
    else:
        points_to_Si_Si3N4 = np.array([0.])


    # keys["corner_inner_r_up"] = keys["corner_r"] - keys["thickness_oxid"]
    # keys["corner_inner_r_low"] = keys["corner_r"] + keys["thickness_oxid"]

    if thickness_oxid_etch_offset == 0:
        points_oxid_x_o = np.array([x2o, x5o, x4o, x3o])
        points_oxid_x = np.array([x2o, x5o, x4o, x3o])
        points_oxid_y_o = np.array([y3o, y3o, y4o, y4o])
        points_oxid_y = np.array([y3o, y3o, y4o, y4o])
        N1 = 1
        dN1 = 2
    else:
        points_oxid_x_o = np.array([x1, x6, x6, x5o, x4o, x3o, x2o, x1])
        points_oxid_x = np.array([x1, x6, x6, x5o, x4o, x3o, x2o, x1])
        points_oxid_y_o = np.array([y2, y2, y3o, y3o, y4o, y4o, y3o, y3o])
        points_oxid_y = np.array([y2, y2, y3o, y3o, y4o, y4o, y3o, y3o])
        N1 = 2
        dN1 = 4
        
    if height_C_etch_offset == 0:
        points_cont_x_o = np.array([x2oc, x5oc, x4oc, x3oc])
        points_cont_y_o = np.array([y3oc, y3oc, y4oc, y4oc])
        points_cont_x = points_cont_x_o.copy()
        points_cont_y = np.array([y3oc, y3oc, y4oc, y4oc])
        N1_C = 1
        dN1_C = 2
    else:
        points_cont_x_o = np.array([x1, x6, x6, x5oc, x4oc, x3oc, x2oc, x1])
        points_cont_y_o = np.array([y2, y2, y3oc, y3oc, y4oc, y4oc, y3oc, y3oc])
        points_cont_x = points_cont_x_o.copy()
        points_cont_y = np.array([y2, y2, y3oc, y3oc, y4oc, y4oc, y3oc, y3oc])
        N1_C = 2
        dN1_C = 4

    if R > 0:
        points_oxid_komplet = do_layers(points_oxid_x_o,points_oxid_x,points_oxid_y_o,points_oxid_y,thickness_oxid_etch_offset=thickness_oxid_etch_offset,R=R,R1=R1,N1 = N1, dN1 = dN1,n=n)
    else:
        if thickness_oxid_etch_offset == 0:
            points_oxid_komplet = np.array([x2o,y3o, x5o,y3o, x4o,y4o, x3o, y4o])
        else:
            points_oxid_komplet = np.array([x1, y2, x6, y2, x6, y3o, x5o, y3o, x4o, y4o, x3o, y4o, x2o, y3o, x1, y3o])
            
    if R > 0:
        points_cont_komplet = do_layers(points_cont_x_o,points_cont_x,points_cont_y_o,points_cont_y,thickness_oxid_etch_offset=height_C_etch_offset,R=R,R1=R1,N1 = N1_C, dN1 = dN1_C,n=n)
    else:
        if height_C_etch_offset == 0:
            points_cont_komplet = np.array([x2,y3oc, x5,y3oc, x4,y4oc, x3, y4oc])
        else:
            points_cont_komplet = np.array([x1, y2, x6, y2, x6, y3oc, x5, y3oc, x4, y4oc, x3, y4oc, x2, y3oc, x1, y3oc])
            
    
    
    if height_etch_offset == 0:
        points_line = np.array([x2, y2, x5, y2, x4, y4, x3, y4])
        N = 1
        dN = 2
    else:
        points_line = np.array([x1, y2, x6, y2, x6, y3, x5, y3, x4, y4, x3, y4, x2, y3, x1, y3])
        N = 2
        dN = 4

    points_line_x_o = points_line[::2].copy()
    points_line_x = points_line[::2].copy()
    points_line_y_o = points_line[1::2].copy()
    points_line_y = points_line[1::2].copy()
    
   
    if R > 0:
        points_line_komplet = do_layers(points_line_x_o,points_line_x,points_line_y_o,points_line_y,thickness_oxid_etch_offset=height_etch_offset,R=R,R1=R1,N1 = N1_C, dN1 = dN1_C,n=n)
    else:
        points_line_komplet = points_line


    return points_substrate, points_line_komplet, points_oxid_komplet, points_cd, points_to_Si_Si3N4, points_cont_komplet



@numba.jit(nopython=True, cache=True, fastmath=True, nogil=True)
def geometry_go_in(cd, R, thickness_oxid, swa, pitch, height, height_offset_substrate, thickness_oxid_Si_Si3N4,
             height_etch_offset, thickness_oxid_etch_offset, height_offset_air,height_C=0, height_C_etch_offset =0, R1 = -1,n=10):
    """
    Calculates all needed point for the jcm layout
    :param cd: width of the grating
    :param R: radius of the corners
    :param thickness_oxid: thickness of the oxide layer
    :param swa: side wall angle
    :param pitch: pitch of the grating
    :param height: height of the grating
    :param height_offset_substrate:  height of the offset of the grating
    :param thickness_oxid_Si_Si3N4: thickness of the oxide layer of the substrate(Si)
    :param height_etch_offset: height of the material that has not been etched
    :param thickness_oxid_etch_offset: height of the oxide in the grooves
    :param height_offset_air: height of the air/vacuum above the grating
    :return:
    """

    if R1 == -1:
        R1 = R
        
    swa_rad = np.deg2rad(swa)
        
    p2 = 0.5 * pitch
    x1 = -0.5 * pitch
    x2 = -0.5 * cd - (height / 2) * np.tan(np.deg2rad(90 - swa))
    x3 = -0.5 * cd + (height / 2) * np.tan(np.deg2rad(90 - swa))
    
    dyo, dxo, dxo_base = calc_shift(thickness_oxid,swa_rad)
    dyoc, dxoc, dxoc_base = calc_shift(thickness_oxid+height_C,swa_rad)
    
    

    #dx = thickness_oxid / np.tan(np.deg2rad((180 - swa) / 2))
    #dxc = (thickness_oxid+height_C) / np.tan(np.deg2rad((180 - swa) / 2))

    x2o = x2 + dxo_base#x2 + dx
    x3o = x3 + dxo#x3 + dx
    x4o = -x3o
    x5o = -x2o
    
    x2oc = x2 + dxoc_base #x2 + dxc
    x3oc = x3 + dxoc #x3 + dxc
    x4oc = -x3oc
    x5oc = -x2oc
    
    #print(x2,x2o,x2oc)

    x4 = -x3
    x5 = -x2
    x6 = -x1

    y1 = -height_offset_substrate - thickness_oxid_Si_Si3N4
    y12 = -thickness_oxid_Si_Si3N4
    y2 = 0.
    y3 = 0. + height_etch_offset
    y3o = 0. + height_etch_offset + thickness_oxid_etch_offset
    y3oc = 0. + height_etch_offset + thickness_oxid_etch_offset + height_C_etch_offset
    
    y4 = height
    y4o = height - dyo
    y4oc = height- dyoc
    y5 = height+ height_offset_air

    points_cd = np.array([-p2, y1, p2, y1, p2, y5, -p2, y5])
    points_substrate = np.array([-p2, y1, p2, y1, p2, y2, -p2, y2])

    if thickness_oxid_Si_Si3N4 > 0:
        points_to_Si_Si3N4 = np.array([-p2, y12, p2, y12, p2, y2, -p2, y2])
    else:
        points_to_Si_Si3N4 = np.array([0.])

    if thickness_oxid_etch_offset == 0:
        points_oxid_x_o = np.array([x2o, x5o, x4o, x3o])
        points_oxid_x = np.array([x2o, x5o, x4o, x3o])
        points_oxid_y_o = np.array([y3o, y3o, y4o, y4o])
        points_oxid_y = np.array([y3o, y3o, y4o, y4o])
        N1 = 1
        dN1 = 2
    else:
        points_oxid_x_o = np.array([x1, x6, x6, x5o, x4o, x3o, x2o, x1])
        points_oxid_x = np.array([x1, x6, x6, x5o, x4o, x3o, x2o, x1])
        points_oxid_y_o = np.array([y2, y2, y3o, y3o, y4o, y4o, y3o, y3o])
        points_oxid_y = np.array([y2, y2, y3o, y3o, y4o, y4o, y3o, y3o])
        N1 = 2
        dN1 = 4
        
    if height_C_etch_offset == 0:
        points_cont_x_o = np.array([x2oc, x5oc, x4oc, x3oc])
        points_cont_y_o = np.array([y3oc, y3oc, y4oc, y4oc])
        points_cont_x = points_cont_x_o.copy()
        points_cont_y = np.array([y3oc, y3oc, y4oc, y4oc])
        N1_C = 1
        dN1_C = 2
    else:
        points_cont_x_o = np.array([x1, x6, x6, x5oc, x4oc, x3oc, x2oc, x1])
        points_cont_y_o = np.array([y2, y2, y3oc, y3oc, y4oc, y4oc, y3oc, y3oc])
        points_cont_x = points_cont_x_o.copy()
        points_cont_y = np.array([y2, y2, y3oc, y3oc, y4oc, y4oc, y3oc, y3oc])
        N1_C = 2
        dN1_C = 4

    if R > 0:
        points_oxid_komplet = do_layers(points_oxid_x_o,points_oxid_x,points_oxid_y_o,points_oxid_y,thickness_oxid_etch_offset=thickness_oxid_etch_offset,R=R,R1=R1,N1 = N1, dN1 = dN1,n=n)
    else:
        if thickness_oxid_etch_offset == 0:
            points_oxid_komplet = np.array([x2o,y3o, x5o,y3o, x4o,y4o, x3o, y4o])
        else:
            points_oxid_komplet = np.array([x1, y2, x6, y2, x6, y3o, x5o, y3o, x4o, y4o, x3o, y4o, x2o, y3o, x1, y3o])
            
    if R > 0:
        points_cont_komplet = do_layers(points_cont_x_o,points_cont_x,points_cont_y_o,points_cont_y,thickness_oxid_etch_offset=height_C_etch_offset,R=R,R1=R1,N1 = N1_C, dN1 = dN1_C,n=n)
    else:
        if height_C_etch_offset == 0:
            points_cont_komplet = np.array([x2,y3oc, x5,y3oc, x4,y4oc, x3, y4oc])
        else:
            points_cont_komplet = np.array([x1, y2, x6, y2, x6, y3oc, x5, y3oc, x4, y4oc, x3, y4oc, x2, y3oc, x1, y3oc])
            
    
    
    if height_etch_offset == 0:
        points_line = np.array([x2, y2, x5, y2, x4, y4, x3, y4])
        N = 1
        dN = 2
    else:
        points_line = np.array([x1, y2, x6, y2, x6, y3, x5, y3, x4, y4, x3, y4, x2, y3, x1, y3])
        N = 2
        dN = 4

    points_line_x_o = points_line[::2].copy()
    points_line_x = points_line[::2].copy()
    points_line_y_o = points_line[1::2].copy()
    points_line_y = points_line[1::2].copy()
    
   
    if R > 0:
        points_line_komplet = do_layers(points_line_x_o,points_line_x,points_line_y_o,points_line_y,thickness_oxid_etch_offset=height_etch_offset,R=R,R1=R1,N1 = N1_C, dN1 = dN1_C,n=n)
    else:
        points_line_komplet = points_line


    return points_substrate, points_line_komplet, points_oxid_komplet, points_cd, points_to_Si_Si3N4, points_cont_komplet


