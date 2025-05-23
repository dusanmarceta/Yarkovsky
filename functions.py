import pygmsh
import numpy as np
from collections import defaultdict
from constants import G, au, speed_of_light, sigma, S0
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import time
from datetime import timedelta


def kepler(e, M, accuracy):
    '''
        Solves Kepler equation using Newton-Raphson method
        for elliptic and hyperbolic orbit depanding on eccentricity
    
        Input:
        e - eccentricity
        M - mean anomaly (rad)
        accuracy - accuracy for Newton-Raphson method (for example 1e-6)
        
        Output:
        E - eccentric (hyperbolic) anomaly (rad)
    '''
    if e > 1:  # hyperbolic orbit (GOODIN & ODELL, 1988)

        L = M / e
        g = 1 / e

        q = 2 * (1 - g)
        r = 3 * L
        s = (np.sqrt(r ** 2 + q ** 3) + r) ** (1 / 3)

        H00 = 2 * r / (s ** 2 + q + (q / s) ** 2)

        #    if np.abs(np.abs(M)-1)<0.01:
        if np.mod(np.abs(M), 0.5) < 0.01 or np.mod(np.abs(M), 0.5) > 0.49:  # numerical problem about this value
            E = (M * np.arcsinh(L) + H00) / (M + 1 + 0.03)  # initial estimate
        else:
            E = (M * np.arcsinh(L) + H00) / (M + 1)  # initial estimate

        delta = 1.0
        while abs(delta) > accuracy:
            f = M - e * np.sinh(E) + E
            f1 = -e * np.cosh(E) + 1
            delta = f / f1
            E = E - delta

    elif e < 1:  # elliptic orbit
        delta = 1.0
        E = M

        while abs(delta) > accuracy:
            f = E - e * np.sin(E) - M
            f1 = 1 - e * np.cos(E)
            delta = f / f1
            E = E - delta

    return E


def ecc2true(E, e):
    '''
    Converts eccentric (or hyperbolic) anomaly to true anomaly
    
    Input:
    E - eccentric (hyperbolic) anomaly (rad)
    e - eccentricity
    
    Output:
    True anomaly (rad)
    '''
    if e > 1:
        return 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(E / 2))
    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)
    
     
def sun_motion(a, e, M0, t):
    """
    calculates orbital postision of an asteroid. Convet time from periapsis to true anomaly
    input:
        a - semi-major axis (au)
        e - eccentricity of the orbit
        t - time since the periapsis (s)
    output:
        x, y - Cartesian coordinates of the Sun in asteroid-centered reference frame (au)
    """
    # mean anomaly
    n=np.sqrt(G/(a*au)**3)
    # mean motion
    M = n*t + M0
    # eccentric anomaly
    E = kepler(e, M, 1e-6)
    
    # Cartesian coordinates of the Sun in asteroid-centered inertial reference frame
    x = a*(np.cos(E)-e) 
    y = a*np.sqrt(1-e**2)*np.sin(E)
    return x, y


def sun_position(axis_lat, axis_long, period, a, e, t, M0, no_motion, precession_period):
    """
    calculates postion of the Sun in the asteroid-fixed (rotating) reference frame
    
    Input:
        axis_lat - latitude of the rotation axis relative to the orbital plane (rad)
        axis_long - longitude of the rotation axis wrt inertial frame measured from the direction of the perihelion (rad)
        period - rotational period of the asteroid
        time - time from the reference epoch (when meridian of the asteroid pointed toward x-axis of the inertial reference frame)
        M0: initial mean anomaly
        initialization: parameter needed for calculating general Yarkovsky effect
    
    output:
        r_sun: unit vector toward the Sun (in asteroid-fixed rotating reference frame)
        r_trans: unit vector in orbital plane normal to r_sun and directed toward direction of motion (in asteroid-fixed rotating reference frame)
        sun_distance: distance to the Sun (au)
        solar irradiance: total irradiance corresponding to sun_distance (W/m2)
    """
    
    
    # instantenous coordinates of the Sun in asteroid-centred inertial reference frame (xOy is orbital plane, x-axis toward pericenter)
    if no_motion == 1: # no motion along the orbit
        x, y = sun_motion(a, e, M0, 0)
    else:
        x, y = sun_motion(a, e, M0, t)
        

    
    r=np.sqrt(x**2+y**2)
    
    nn = np.array([0,0,1]) # normal to orbital plane
    
    [xt, yt, zt] = np.cross(nn, np.array([x/r, y/r, 0])) # unit vector toward the Sun
        
    if precession_period is not np.inf:
        axis_long -= 2*np.pi * t / precession_period
    
    # 1. we rotate about z axis for angle axis_long
    R1 = np.array([[np.cos(axis_long), -np.sin(axis_long), 0],
                    [np.sin(axis_long), np.cos(axis_long), 0],
                    [0, 0, 1]]);
    
    [x1, y1, z1] = np.matmul(R1, [x/r, y/r, 0])
    [x1t, y1t, z1t] = np.matmul(R1, [xt, yt, zt])
    # 2. we rotate about y axis for angle (pi/2 - axis_lat)
    y_angle = -(np.pi/2 - axis_lat)
    
    R2 = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                    [0, 1, 0],
                    [-np.sin(y_angle), 0, np.cos(y_angle)]]);
    
    [x2, y2, z2] = np.matmul(R2, [x1, y1, z1])
    [x2t, y2t, z2t] = np.matmul(R2, [x1t, y1t, z1t])
    
    # 3. we rotate about z axis for rotation angle
    rotation_angle = -(2*np.pi/period * t)
    R3 = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                    [0, 0, 1]]);
    
    [x3, y3, z3] = np.matmul(R3, [x2, y2, z2])
    [x3t, y3t, z3t] = np.matmul(R3, [x2t, y2t, z2t])
    
    # rotation from inertial to asteroid-fixed frame
    R=np.linalg.inv(np.matmul(np.matmul(R3, R2), R1))
    
    [xs, ys, zs] = np.matmul(R, np.array([x, y, 0])) # coordinates of the sun in asteroid_fixed reference frame (au)
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    
    solar_irradiance = 1361./rs**2 # total solar irradiance at distance rs

    return([x3, y3, z3], [x3t, y3t, z3t], rs, solar_irradiance)
    

def layers(D, x1, N):
    
    if D/x1 < N: # if the first section is to large so that other must be decreased to reach N sections
        # we set equidistant division so that the section size is smaller than the required first section x1
        
        N = int(np.ceil(D/x1)) + 1 
        
        points = np.linspace(0, D, N)
    
    else:
        # Funkcija koja vraća grešku za dati r
        def error(r):
            return x1 * (r**N - 1) / (r - 1) - D
    
        # Početna procena za r
        initial_guess = 1.1
        
        # Izračunaj r pomoću fsolve (Newton-Raphson metoda)
        r_solution = fsolve(error, initial_guess)[0]
        
        
        
        # Generiši korake koristeći izračunat r
        steps = x1 * r_solution ** np.arange(N)
        
        # Podesi korake tako da njihov zbir bude tačno D
        steps *= D / np.sum(steps)
    
        # Kreiraj tačke podele
        points = np.concatenate(([0], np.cumsum(steps)))
    
    return points[1:]




def ellipsoid_area(a, b, c):
    '''
    Knud Thomsen's formula
    '''
    p = 1.6075
    return 4 * np.pi * (( (a*b)**p + (a*c)**p + (b*c)**p ) / 3)**(1/p)

def euclid_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def triangle_area(P1, P2, P3):
        
    # Calculate the vectors
    v1 = P2 - P1
    v2 = P3 - P1
    
    # Compute the cross product of v1 and v2
    cross_product = np.cross(v1, v2)
    
    # Compute the magnitude of the cross product
    return  0.5 * np.linalg.norm(cross_product)


def triangle_normal(P1, P2, P3):
        
    # Calculate the vectors
    v1 = P2 - P1
    v2 = P3 - P1
    
    # Compute the cross product of v1 and v2
    cross_product = np.cross(v1, v2)
    
    # unite normal vector
    normal = cross_product / np.linalg.norm(cross_product)
    
    # Check the direction of the normal vector
    # Vector from origin to P1
    origin_to_P1 = P1
    
    # If the normal points towards the origin, flip it
    if np.dot(normal, origin_to_P1) > 0:
        normal = -normal
    
    return normal


def frustum_volume(S1, S2, h):
    '''
    volume formula for a frustum of a prism if the two bases are not identical
    '''
    return h*(S1 + S2 + np.sqrt(S1 * S2))/3

def facet_centroid(p1, p2, p3):

    # Calculate the centroid
    return  (p1 + p2 + p3) / 3


# Function to compute normal vectors for an ellipsoid
def compute_normals(points, a, b, c):
    normals = np.zeros_like(points)
    normals[:, 0] = (points[:, 0] / a**2)
    normals[:, 1] = (points[:, 1] / b**2)
    normals[:, 2] = (points[:, 2] / c**2)
    norms = np.linalg.norm(normals, axis=1)
    return normals / norms[:, np.newaxis]




# Function to find rows that share exactly two values
def find_matching_rows(arr, pair_dict):
    match = []
    for idx, row in enumerate(arr):
#        print(f"Row {idx} ({row}):")
        
        temp = []
        for i in range(3):
            for j in range(i+1, 3):
                val1, val2 = row[i], row[j]
                pair = tuple(sorted([val1, val2]))
                matching_rows = [r for r in pair_dict[pair] if r != idx]
#                print(f"  Values {val1}, {val2}: Rows {matching_rows}")
                temp.append(matching_rows[0])
#        print('treuntna = ', trenutna)
        match.append(temp)
    return(match)


def prismatic_mesh(a, b, c, facet_size, layers):
    '''
    The ellipsoid is devided into layers with depths bellow surface defined by array layers.
    Every layer is meshed with triangular prism. The height of triangula bases is defined by facet_size.
    inputs:
        a, b, c - semi-axes of the ellipsoid
        facet_size - dimension of the surface mesh element
        layers - depths of the layers
    output:
    
    ''' 
    thickness = np.diff(layers)
    thickness = np.insert(thickness, 0, layers[0])

    # Create a geometry and define a sphere with controlled mesh size
    with pygmsh.geo.Geometry() as geom:
        # Define a sphere with radius 1 and control the mesh size
    #    sphere = geom.add_ball([0, 0, 0], 1.0, mesh_size=triangle_size)
        ellipsoid = geom.add_ellipsoid([0, 0, 0], [a, b, c], mesh_size = facet_size)
    
        # Generate the mesh
        mesh = geom.generate_mesh()
    
    # Extract points and faces
    points_outer = mesh.points # coordinates of the surface points
    
    
    facets = mesh.get_cells_type("triangle") - 1 # indices of the surface facets
    
    # for some reason there is also a point [0,0,0] so this is to remove it
    points_outer = points_outer[1:]
    
    
    # array of all points of the grid. Every array corresponds to one layer, starting from the surface
    points  = np.zeros([len(layers) + 1, len(points_outer), 3])
    
    # surface points
    points[0] = points_outer
    
    # Compute normals for the outer ellipsoid
    normals = compute_normals(points_outer, a, b, c)
    
    # points of the internal layers
    for i, hh in enumerate(layers):
    
        points[i+1] = points_outer - hh * normals
        
    # =============================================================================
    # finding neighboring facets (works!!!)
    # =============================================================================
    pair_dict = defaultdict(list)
    
    # Populate the dictionary with pairs from all rows
    for idx, row in enumerate(facets):
        for i in range(3):
            for j in range(i+1, 3):
                pair = tuple(sorted([row[i], row[j]]))  # Sort to avoid (a,b) and (b,a) mismatch
                pair_dict[pair].append(idx)
    
    # Call the function
    neighbor_facets = find_matching_rows(facets, pair_dict) # WORKS PERFECTLY! (double checked)

    # =============================================================================
    # for every cell this is the list of neighboring cells in the format
    # [above, same level, same level, same level, bellow] (works!!!) double checked
    # =============================================================================
    fictuous_cell = int(len(facets) * len(layers)) # This is a fictitious cell that we use as a neighbor to the surface cells and the cells of the deepest layer
    neighbor_cells = []
    br = -1
    for i in range(len(layers)):
        for j in range(len(facets)):
            
            temp = []
            br += 1

            if i == 0: # surface cells 
                temp.append(fictuous_cell) # above
                temp.extend(np.array(neighbor_facets[j]) + i * len(facets)) # same level
                temp.append((i+1) * len(facets) +  j) # bellow
                
            elif i > 0 and i < len(layers) - 1: # inside (but not next to central)
                
                temp.append((i-1) * len(facets) +  j) # above
                temp.extend(np.array(neighbor_facets[j]) + i * len(facets)) # same level
                temp.append((i+1) * len(facets) +  j) # below
                
            else: # above central cell
                
                temp.append((i-1) * len(facets) +  j) # above
                temp.extend(np.array(neighbor_facets[j]) + i * len(facets)) # same level
                temp.append(fictuous_cell) # below (central cell)
            
            neighbor_cells.append(temp)
            
    neighbor_cells.append([fictuous_cell, fictuous_cell, fictuous_cell, fictuous_cell, fictuous_cell]) # za fiktivnu celiju ona je sama sebi susedna
            
    # =============================================================================
    # Cell areas (Works!) 
    # =============================================================================
    facets_area = np.zeros([len(points), len(facets)])
    
    for i in range(len(points)-1):
        for j in range(len(facets)):
            facets_area[i][j] = triangle_area(points[i][facets][j][0], points[i][facets][j][1], points[i][facets][j][2])
    
    surface_normals = np.zeros([len(facets), 3])      
    for i in range(len(facets)): 
        surface_normals[i] = triangle_normal(points[0][facets][i][0], points[0][facets][i][1], points[0][facets][i][2])       
    
    br = -1
    areas = np.zeros([len(neighbor_cells), 5]) # the areas of the shared sides with neighboring cells
    for i in range(len(points)-1):
        for j in range(len(facets)):
            br += 1
            # SIDE 1
            side = neighbor_cells[br][1] # redni broj celije na mestu 2 (u istom nivou)
            in_line = side - i*len(facets) # indeks susedne celije 2 (u istom nivou)
            
            # zajednicke tacke
            common_points = np.intersect1d(facets[j], facets[in_line])
            base_up = euclid_distance(points[i][common_points[0]], points[i][common_points[1]])
            base_down = euclid_distance(points[i+1][common_points[0]], points[i+1][common_points[1]])
            
            if i==0:
                area_1 = (base_up + base_down) * layers[i]/2
            else:
                area_1 = (base_up + base_down) * (layers[i]-layers[i-1])/2
                     
            # SIDE 2
            side = neighbor_cells[br][2] # redni broj celije na mestu 3 (u istom nivou)
            in_line = side - i*len(facets) # indeks susedne celije 2 (u istom nivou)
            
            # zajednicke tacke
            common_points = np.intersect1d(facets[j], facets[in_line])
            base_up = euclid_distance(points[i][common_points[0]], points[i][common_points[1]])
            base_down = euclid_distance(points[i+1][common_points[0]], points[i+1][common_points[1]])
            
            if i==0:
                area_2 = (base_up + base_down) * layers[i]/2
            else:
                area_2 = (base_up + base_down) * (layers[i]-layers[i-1])/2
                
            # SIDE 3
            side = neighbor_cells[br][3] # redni broj celije na mestu 4 (u istom nivou)
            in_line = side - i*len(facets) # indeks susedne celije 2 (u istom nivou)
            
            # zajednicke tacke
            common_points = np.intersect1d(facets[j], facets[in_line])
            base_up = euclid_distance(points[i][common_points[0]], points[i][common_points[1]])
            base_down = euclid_distance(points[i+1][common_points[0]], points[i+1][common_points[1]])
            
            if i==0:
                area_3 = (base_up + base_down) * layers[i]/2
            else:
                area_3 = (base_up + base_down) * (layers[i]-layers[i-1])/2
                
            # upper
            areas[br][0] = facets_area[i][j]
                
            # lower
            areas[br][4] = facets_area[i+1][j]
            
            # in the level
            areas[br][1], areas[br][2], areas[br][3] = area_1, area_2, area_3
            
    # =============================================================================
    # Distances
    # =============================================================================
    br = -1
    distances = np.zeros([len(neighbor_cells), 5])
    for i in range(len(points)-1):
        for j in range(len(facets)):
            br += 1
    
            c0_up = facet_centroid(points[i][facets[j][0]], points[i][facets[j][1]], points[i][facets[j][2]]) # centroid of the current cell (upper face)
            c0_down = facet_centroid(points[i+1][facets[j][0]], points[i+1][facets[j][1]], points[i+1][facets[j][2]]) # centroid of the current cell (lower face)

            # SIDE 1
            temp = neighbor_cells[br][1] # redni broj celije na mestu 2 (u istom nivou)
            in_line = temp - i*len(facets) # indeks susedne celije 2 (u istom nivou)   
            c1_up = facet_centroid(points[i][facets[in_line][0]], points[i][facets[in_line][1]], points[i][facets[in_line][2]]) # centroid of the neighbour cell
            c1_down = facet_centroid(points[i+1][facets[in_line][0]], points[i+1][facets[in_line][1]], points[i+1][facets[in_line][2]]) # centroid of the neighbour cell
            distance_up = euclid_distance(c0_up, c1_up)
            distance_down = euclid_distance(c0_down, c1_down)
            distances[br][1] = (distance_up + distance_down)/2 # distance between centroids
            
            # SIDE 2
            temp = neighbor_cells[br][2] # redni broj celije na mestu 2 (u istom nivou)
            in_line = temp - i*len(facets) # indeks susedne celije 2 (u istom nivou)   
            c1_up = facet_centroid(points[i][facets[in_line][0]], points[i][facets[in_line][1]], points[i][facets[in_line][2]]) # centroid of the neighbour cell
            c1_down = facet_centroid(points[i+1][facets[in_line][0]], points[i+1][facets[in_line][1]], points[i+1][facets[in_line][2]]) # centroid of the neighbour cell
            distance_up = euclid_distance(c0_up, c1_up)
            distance_down = euclid_distance(c0_down, c1_down)
            distances[br][2] = (distance_up + distance_down)/2 # distance between centroids
            
            # SIDE 3
            temp = neighbor_cells[br][3] # redni broj celije na mestu 2 (u istom nivou)
            in_line = temp - i*len(facets) # indeks susedne celije 2 (u istom nivou)   
            c1_up = facet_centroid(points[i][facets[in_line][0]], points[i][facets[in_line][1]], points[i][facets[in_line][2]]) # centroid of the neighbour cell
            c1_down = facet_centroid(points[i+1][facets[in_line][0]], points[i+1][facets[in_line][1]], points[i+1][facets[in_line][2]]) # centroid of the neighbour cell
            distance_up = euclid_distance(c0_up, c1_up)
            distance_down = euclid_distance(c0_down, c1_down)
            distances[br][3] = (distance_up + distance_down)/2 # distance between centroids
            
            # above
            if i==0:
                distances[br][0] = np.inf
            else:
                distances[br][0] = (thickness[i] + thickness[i-1])/2
                 
            # below
            if i == len(layers)-1: # the deepest layer
                distances[br][4] = np.inf
            else:
                distances[br][4] = (thickness[i] + thickness[i+1])/2
                
    distances[-1] = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]) # fictuous cell
        
    # =============================================================================
    # Cell volumes (works !!!)
    # =============================================================================
    volumes = np.zeros(len(neighbor_cells))    
    for i in range(len(neighbor_cells)-1): # -1 because we don't calculate for the central cell
        level = i // len(facets)
        in_level = np.mod(i, len(facets))
        base_1 = facets_area[level][in_level]
        base_2 = facets_area[level + 1][in_level]
        
        if level==0:
            volumes[i] = frustum_volume(base_1, base_2, layers[level])
        else:
            volumes[i] = frustum_volume(base_1, base_2, layers[level] - layers[level-1])
        
    volumes[-1] = np.inf # fictuous cell
            
    return (neighbor_cells, volumes, areas, distances, np.arange(len(facets)), facets_area[0], surface_normals)

def seasonal_yarkovsky_effect(semi_axis_a, semi_axis_b, semi_axis_c,  # shape of the asteroid
                              rho, k, albedo, cp, eps,  # physical characteristics
                              axis_lat, axis_long, rotation_period, precession_period,  # rotation state
                              semi_major_axis, eccentricity,  # orbital elements
                              facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor,  # numerical grid parameters
                              progress_file): # file where the estimated remaining calculation time is written periodically



    # mean motion of the asteroid (rad/s)
    mean_motion=np.sqrt(G/(semi_major_axis*au)**3)
    
    # orbital period of the asteroid (s)
    orbital_period = 2*np.pi / mean_motion
    
    # Depth of the seasonal thermal wave penetration
    ls = np.sqrt(k/rho/cp/mean_motion)
    
    # total depth of the numerical grid
    total_depth = number_of_thermal_wave_depths * ls
    
    # We ensure that the total depth does not exceed 80% of the shortest semi-axis
    total_depth  = np.min([0.8 * semi_axis_a, 0.8 * semi_axis_b, 0.8 * semi_axis_c, total_depth])
    
    # depth of the first layer
    first_layer_depth_abs = np.min([ls, total_depth/number_of_thermal_wave_depths]) * first_layer_depth
    
    # depths of all layers
    layer_depths  = layers(total_depth, first_layer_depth_abs, number_of_layers)
    

    # generating the mesh
    neighbor_cells, volumes, areas, distances, surface_cells, surface_areas, surface_normals = prismatic_mesh(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layer_depths)

    # initial setting for the time step (it will usually be much smaller than this because it depands on the smallest distance between the cells)
    time_step = rotation_period / 24 
    
    # Minimum distance between neighboring cells
    d_min = np.min(distances)
    
    # critical time step that depands on the minimum distance between the cells
    dt_critical=d_min**2*rho*cp/k
    
    # chosing the smaller time step between 2 options
    time_step = np.min([time_step, dt_critical/time_step_factor])

    
    # mass of the asteorid (kg)
    mass = 4/3 * semi_axis_a * semi_axis_b * semi_axis_c * np.pi * rho
    
    # Equilibrium temperature assuming a heliocentric distance equal to the semi-major axis
    T_equilibrium = (S0/semi_major_axis**2/4/sigma)**0.25
    
    # Setting initial temepratures of all cells to be equal to T_equilibrium
    T = np.zeros(len(volumes)) + T_equilibrium # temperature svih celija
    
    # Array to store last 2 values of the Yarkovsky drift (to check for divergence)
    drift = [0, 0]

    # iteration counter
    i=0
    
    # total time
    total_time = 0 
    
    # for mesuring of execution time
    time_1 = time.time() 
    
    total_number_of_iterations = orbital_period / time_step
    
    total_drift = 0
    
    drift_za_plot = []
    
    while total_time <= orbital_period:
        
        drift[0] = drift[1]
                
        # position of the Sun in asteroid-fixed reference frame
        r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat = axis_lat, axis_long = axis_long, 
                                                                      period = rotation_period, 
                                                                      a = semi_major_axis, 
                                                                      e = eccentricity, 
                                                                      t = total_time, 
                                                                      M0 = np.pi, # starting from aphelion 
                                                                      no_motion = 0, # not needed for isolated seasonal effect
                                                                      precession_period = precession_period) 
   
        # temperature difference relative to neighboring cells
        delta_T = T[neighbor_cells] - T[:, None] 
        
        # Heat flux due to conduction
        J = np.sum(k * delta_T / distances * areas, axis=1)
            
        # External heat flux
        J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(surface_normals, r_sun), 0)) - sigma*eps*(T[surface_cells])**4)
        
        # Temeprature change
        dTdt = J/(rho * volumes * cp)
        
        # Updated temperature
        T += dTdt * time_step
    
        # Total thermal force
        F=2/3*eps*sigma/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
        
        # Transfersal thermal force (Yarkovsky force)
        B = np.dot(F, r_trans)
    
        # Semi-major axis drift
        dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
        
        drift[1] = dadt
        
        

        if np.abs(np.nanmax(drift) - np.nanmin(drift)) > 1e2: # divergence due to the large time step
            
            # Divergence happened. Time step is decreased and the entire calculation starts again.
            total_time = 0
            i = 0
            time_step = time_step / 2
            T = np.zeros(len(volumes)) + T_equilibrium
            drift = [0, 0]
            total_number_of_iterations = orbital_period / time_step
            total_drift = 0
            time_1 = time.time() # for measuring of the execution time
            
        else:
            
            total_time += time_step
            i += 1
            total_drift += dadt
            drift_za_plot.append(dadt)
            
        if np.mod(i, 100)==0 and i > 0:
            
            per_iteration = (time.time() - time_1)/i
            estimated_time = (total_number_of_iterations - i) * per_iteration
            hours, remainder = divmod(int(round(estimated_time)), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = '{:02}:{:02}:{:02}'.format(hours, minutes, seconds)
#            print('Estimated execution time is {}'.format(formatted_time))
            
            
#            print('{}, {}, {}, {}, {}, {}'.format(facet_size, number_of_thermal_wave_depths, np.round(first_layer_depth/ls, 2), number_of_layers, time_step_factor, formatted_time))
            zapis = 'facet_size = {} m\nnumber of thermal wave depths = {}\nfirst layer depth = {} m\nnumber of layers = {}\ntime step factor = {}\nestimated calculation time remaining = {}'.format(np.round(facet_size, 3), number_of_thermal_wave_depths, np.round(first_layer_depth_abs, 3), number_of_layers, time_step_factor, formatted_time)
            np.savetxt(progress_file, [zapis], fmt='%s')
        

    return total_drift/total_number_of_iterations, drift_za_plot, total_time


def diurnal_yarkovsky_effect(semi_axis_a, semi_axis_b, semi_axis_c, # shape of the asteroid
                             rho, k, albedo, cp, eps, # physical characteristics
                             axis_lat, axis_long, rotation_period, precession_period, # rotation state
                             semi_major_axis, eccentricity, number_of_locations, # orbit
                             facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor, # numerical grid parameters
                             max_tol, min_tol, mean_tol, amplitude_tol, maximum_number_of_rotations): # convergence parameters

    # Depth of the diurnal thermal wave penetration
    ls = np.sqrt(k*rotation_period/rho/cp/(2*np.pi))
    
    # total depth of the numerical grid
    total_depth = number_of_thermal_wave_depths * ls
    
    # depth of the first layer
    first_layer_depth = ls * first_layer_depth
    
    # depths of all layers
    layer_depths  = layers(total_depth, first_layer_depth, number_of_layers)
   
    # generating the mesh
    neighbor_cells, volumes, areas, distances, surface_cells, surface_areas, surface_normals = prismatic_mesh(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layer_depths)

    
    # coorcinates of the surface cells
    lat_surface = np.rad2deg(np.arcsin(np.transpose(surface_normals)[2]))# latitude
    long_surface = np.rad2deg(np.arctan2(np.transpose(surface_normals)[1], np.transpose(surface_normals)[0])) # longitude

    grid_lon, grid_lat = np.meshgrid(
    np.linspace(-180, 180, 3600),  # 1° razmak
    np.linspace(min(lat_surface), max(lat_surface), 1800)  # 0.1° razmak
    )
    
    
    


    
    
    # initial setting for the time step (it will usually be much smaller than this because it depands on the smallest distance between the cells)
    time_step = rotation_period / 24 
    
    # Minimum distance between neighboring cells
    d_min = np.min(distances)
    
    # critical time step that depands on the minimum distance between the cells
    dt_critical=d_min**2*rho*cp/k
    
    # chosing the smaller time step between 2 options
    time_step = np.min([time_step, dt_critical/time_step_factor])
    
    # To ensure an integer number of steps per rotation
    number_of_steps_per_rotation = np.ceil(rotation_period / time_step)
    time_step = rotation_period/number_of_steps_per_rotation 

    # mean motion of the asteroid (rad/s)
    mean_motion=np.sqrt(G/(semi_major_axis*au)**3)
        
    # mass of the asteorid (kg)
    mass = 4/3 * semi_axis_a * semi_axis_b * semi_axis_c * np.pi * rho
    
    
    
    
    
    # A different convergence criterion is used for spherical asteroid
    sphere = 0 # Flag indicating whether the asteroid is a perfect sphere
    if semi_axis_a == semi_axis_b and semi_axis_a == semi_axis_c and precession_period is np.inf:
        sphere = 1
    
    # Array to store Yarkovsky drift values at all orbital positions (the first and last element correspond to perihelion, so we calculate only one)
    drift_for_location = np.zeros(number_of_locations + 1) 
    
    # Array to store mean anomaly (the first and last element correspond to perihelion)
    M_for_location = np.linspace(0, 2 * np.pi, number_of_locations + 1)
    
    T_surface = np.zeros([number_of_locations, 1800, 3600])
    T_equator_rotation = np.zeros([number_of_locations, int(number_of_steps_per_rotation)])
    T_noon = np.zeros([number_of_locations, number_of_layers])
    T_midnight = np.zeros([number_of_locations, number_of_layers])
    
    
    drift_evolution = []

    for location in range(number_of_locations):
        
        np.savetxt('progres.txt', [location])
        
        print('{}, {}, {}, {}, {}, Location no. {} out of {}'.format(facet_size, number_of_thermal_wave_depths, np.round(first_layer_depth/ls, 2), number_of_layers, time_step_factor, location + 1, number_of_locations))
        
        
        
        # Mean anomaly at the current position in the orbit
        M = M_for_location[location]
        
        # position of the Sun in asteroid-fixed reference frame
        r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat = axis_lat, axis_long = axis_long, 
                                                                      period = rotation_period, 
                                                                      a = semi_major_axis, 
                                                                      e = eccentricity, 
                                                                      t = 0, 
                                                                      M0 = M, 
                                                                      no_motion = 1, # not needed for isolated seasonal effect
                                                                      precession_period = precession_period)
        
        
        long_midnight = np.mod(np.rad2deg(np.arctan2(r_sun[1], r_sun[0])), 360) + 180
                
        long_noon = np.mod(np.rad2deg(np.arctan2(r_sun[1], r_sun[0])), 360)
        
        if long_midnight > 180:
            long_midnight = long_midnight - 360
        elif long_midnight < -180:
            long_midnight = long_midnight + 360
            
        if long_noon > 180:
            long_noon = long_noon - 360
        elif long_noon < -180:
            long_noon = long_noon + 360
    
    
    
        dist_midnight = np.sqrt((long_surface - long_midnight)**2 + (lat_surface - 0)**2)
        idx_midnight = np.argmin(dist_midnight)
        
        dist_noon = np.sqrt((long_surface - long_noon)**2 + (lat_surface - 0)**2)
        idx_noon = np.argmin(dist_noon)
        
        idx_noon_depth = np.arange(idx_noon, number_of_layers * len(surface_areas) + idx_noon, len(surface_areas))
        idx_midnight_depth = np.arange(idx_midnight, number_of_layers * len(surface_areas) + idx_midnight, len(surface_areas))


        
        long_surface_location = long_surface - np.mod(np.rad2deg(np.arctan2(r_sun[1], r_sun[0])), 360)
        
        long_surface_location = ((long_surface_location + 180) % 360) - 180
        
        
        long_extended = np.concatenate([
        long_surface_location - 360,
        long_surface_location,
        long_surface_location + 360
        ])
    
    
        lat_extended = np.tile(lat_surface, 3)
        
        # Equilibrium temperature assuming a heliocentric distance equal to the semi-major axis
        T_equilibrium = (S0/sun_distance**2/4/sigma)**0.25
            
        # Setting initial temepratures of all cells to be equal to T_equilibrium
        T = np.zeros(len(volumes)) + T_equilibrium # temperature svih celija
        
        # List for storing Yarkovsky drift values
        drift = []
        T_equator_rotation_location = []
#        T_equatorial_midnight = []
        
        # total time
        total_time = 0.
        
        # iteration counter
        i=0
        
        long_midnight = 180
        long_noon = 0
        while True:
            
            # position of the Sun in asteroid-fixed reference frame
            r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat = axis_lat, axis_long = axis_long, 
                                                                          period = rotation_period, 
                                                                          a = semi_major_axis, 
                                                                          e = eccentricity, 
                                                                          t = total_time, 
                                                                          M0 = M, 
                                                                          no_motion = 1, # not needed for isolated seasonal effect
                                                                          precession_period = precession_period) 
#            if total_time == 0:
#                long_midnight = np.mod(np.rad2deg(np.arctan2(r_sun[1], r_sun[0])), 360) + 180
#                
#                long_noon = np.mod(np.rad2deg(np.arctan2(r_sun[1], r_sun[0])), 360)
#                
#                if long_midnight > 180:
#                    long_midnight = long_midnight - 360
#                elif long_midnight < -180:
#                    long_midnight = long_midnight + 360
#                    
#                if long_noon > 180:
#                    long_noon = long_noon - 360
#                elif long_noon < -180:
#                    long_noon = long_noon + 360
#            
#            
#            
#                dist_midnight = np.sqrt((long_surface - long_midnight)**2 + (lat_surface - 0)**2)
#                idx_midnight = np.argmin(dist_midnight)
#                
#                dist_noon = np.sqrt((long_surface - long_noon)**2 + (lat_surface - 0)**2)
#                idx_noon = np.argmin(dist_noon)
#                
#                idx_noon_depth = np.arange(idx_noon, number_of_layers * len(surface_areas) + idx_noon, len(surface_areas))
#                idx_midnight_depth = np.arange(idx_midnight, number_of_layers * len(surface_areas) + idx_midnight, len(surface_areas))
#            
            

            

            # temperature difference relative to neighboring cells
            delta_T = T[neighbor_cells] - T[:, None] 
            
            # Heat flux due to conduction
            J = np.sum(k * delta_T / distances * areas, axis=1)
                
            # External heat flux
            J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
             np.dot(surface_normals, r_sun), 0)) - sigma*eps*(T[surface_cells])**4)
            
            # Temeprature change
            dTdt = J/(rho * volumes * cp)
            
            # Updated temperature
            T += dTdt * time_step
        
            # Total thermal force
            F=2/3*eps*sigma/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
            
            # Transfersal thermal force (Yarkovsky force)
            B = np.dot(F, r_trans)
        
            # Semi-major axis drift
            dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
            
            
            drift.append(dadt)
            
            T_equator_rotation_location.append(T[surface_cells][idx_midnight])
#            T_equatorial_noon.append(T[surface_cells][idx_midnight])
            

            if np.abs(np.nanmax(drift) - np.nanmin(drift)) > 1e2: # divergence due to the large time step
                
                # Divergence happened. Time step is decreased and the entire calculation starts again.
                total_time = 0
                i = 0
                time_step = time_step / 2
                T = np.zeros(len(volumes)) + T_equilibrium
                drift = []
                
            else:
                
                total_time += time_step
                i += 1
    
            # Checking for convergence
            if np.mod(i, number_of_steps_per_rotation ) == 0 and i >= 2*number_of_steps_per_rotation: # Full rotation completed (starting from the second rotation)
                
                
                # number of full rotations
                number_of_rotations = i / number_of_steps_per_rotation 
                
                drift_1 = drift[-int(2*number_of_steps_per_rotation):-int(number_of_steps_per_rotation)] # Drift values from the rotation preceding the last one
                drift_2 = drift[-int(number_of_steps_per_rotation):] # Drift values from the last rotation
                
                # Extreme values during the last two rotations
                max_value_1 = np.max(drift_1)
                min_value_1 = np.min(drift_1)
                max_value_2 = np.max(drift_2)
                min_value_2 = np.min(drift_2)
                
                # Amplitude during the final rotation
                amplitude = (max_value_2 - min_value_2)/ np.mean(drift_2)
                
                # Difference between the maxima during the last two rotations
                diff_max = np.abs((max_value_2 - max_value_1)/max_value_2) 
                
                # Difference between the minima during the last two rotations
                diff_min = np.abs((min_value_2 - min_value_1)/min_value_2)
                
                # Difference between the means during the last two rotations
                diff_mean = np.abs((np.mean(drift_2) - np.mean(drift_1))/np.mean(drift_1))
                    
                if diff_max < max_tol and diff_min < min_tol and diff_mean < mean_tol and amplitude < amplitude_tol and sphere == 1: # for a spherical asteroid
                    # Storing the mean drift value from the final rotation
                    drift_for_location[location] = np.mean(drift_2)

                    break # next position in the orbit
        
                elif diff_max < max_tol and diff_min < min_tol and diff_mean < mean_tol and sphere == 0: # for a ellipsoidal asteroid
                    # Storing the mean drift value from the final rotation
                    drift_for_location[location] = np.mean(drift_2)

                    break # next position in the orbit
                    
                elif number_of_rotations >= maximum_number_of_rotations: # to prevent too long simulation
                    drift_for_location[location] = np.mean(drift_2)

                    break # next position in the orbit
        
        T_extended = np.tile(T[surface_cells], 3)
        T_surface[location] = griddata((long_extended, lat_extended), T_extended, (grid_lon, grid_lat), method='cubic')

#        grid_lat_full = np.concatenate([[-90], grid_lat, [90]])
        
        
        
        drift_evolution.append(drift)
        
                    
        T_equator_rotation[location] = T_equator_rotation_location[-int(number_of_steps_per_rotation):]
        T_noon[location] = T[idx_noon_depth]
        T_midnight[location] = T[idx_midnight_depth]
     
    # the last vvalue also corresponds to perihelion so we do not need to calculate it again
    drift_for_location[-1] = drift_for_location[0]
        
    # Mean value of the Yerkovsky effect with respect to time for the entire orbit
    total_effect = np.trapz(drift_for_location, M_for_location)/(2*np.pi)
    

    return total_effect, drift_evolution, drift_for_location, M_for_location, T_equator_rotation, T_noon, T_midnight, layer_depths,  grid_lon, grid_lat, T_surface


def general_yarkovsky_effect(semi_axis_a, semi_axis_b, semi_axis_c, # shape of the asteroid
                             rho, k, albedo, cp, eps, # physical characteristics
                             axis_lat, axis_long, rotation_period, precession_period, # rotation state
                             semi_major_axis, eccentricity, number_of_locations, # orbit
                             facet_size, number_of_thermal_wave_depths, first_layer_depth, number_of_layers, time_step_factor, # numerical grid parameters
                             max_tol, min_tol, mean_tol, amplitude_tol, maximum_number_of_rotations): # convergence parameters

    initialization = 1
    
    # mean motion of the asteroid (rad/s)
    mean_motion=np.sqrt(G/(semi_major_axis*au)**3)
    
    # orbital period of the asteroid (s)
    orbital_period = 2*np.pi / mean_motion
    
    # Depth of the diurnal thermal wave penetration
    l_diurnal = np.sqrt(k*rotation_period/rho/cp/(2*np.pi))
    
    # Depth of the seasonal thermal wave penetration
    l_seasonal = np.sqrt(k/rho/cp/mean_motion)
    
    # total depth of the numerical grid
    total_depth = number_of_thermal_wave_depths * np.max([l_diurnal, l_seasonal])
     
    # We ensure that the total depth does not exceed 80% of the shortest semi-axis
    total_depth  = np.min([0.8 * semi_axis_a, 0.8 * semi_axis_b, 0.8 * semi_axis_c, total_depth])
    
    # depth of the first layer
    first_layer_depth = l_diurnal * first_layer_depth
    
    # depths of all layers
    layer_depths  = layers(total_depth, first_layer_depth, number_of_layers)
   
    # generating the mesh
    neighbor_cells, volumes, areas, distances, surface_cells, surface_areas, surface_normals = prismatic_mesh(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layer_depths)

    # initial setting for the time step (it will usually be much smaller than this because it depands on the smallest distance between the cells)
    time_step = rotation_period / 24 
    
    # Minimum distance between neighboring cells
    d_min = np.min(distances)
    
    # critical time step that depands on the minimum distance between the cells
    dt_critical=d_min**2*rho*cp/k
    
    # chosing the smaller time step between 2 options
    time_step = np.min([time_step, dt_critical/time_step_factor])
    
    # To ensure an integer number of steps per rotation
    number_of_steps_per_rotation = np.ceil(rotation_period / time_step)
    time_step = rotation_period/number_of_steps_per_rotation 

    
        
    # mass of the asteorid (kg)
    mass = 4/3 * semi_axis_a * semi_axis_b * semi_axis_c * np.pi * rho
    
    # Equilibrium temperature assuming a heliocentric distance equal to the semi-major axis
    T_equilibrium = (S0/semi_major_axis**2/4/sigma)**0.25
    
    # A different convergence criterion is used for spherical asteroid
    sphere = 0 # Flag indicating whether the asteroid is a perfect sphere
    if semi_axis_a == semi_axis_b and semi_axis_a == semi_axis_c and precession_period is np.inf:
        sphere = 1
    
    # Setting initial temepratures of all cells to be equal to T_equilibrium
    T = np.zeros(len(volumes)) + T_equilibrium # temperature svih celija
    
    # List for storing Yarkovsky drift values
    drift = []
    
    # total time
    total_time = 0.
    
    # iteration counter
    i=0
    
    print("Initialization started")
    while initialization == 1:
        
        
        # position of the Sun in asteroid-fixed reference frame
        r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat = axis_lat, axis_long = axis_long, 
                                                                      period = rotation_period, 
                                                                      a = semi_major_axis, 
                                                                      e = eccentricity, 
                                                                      t = total_time, 
                                                                      M0 = np.pi, 
                                                                      no_motion = initialization, # not needed for isolated seasonal effect
                                                                      precession_period = precession_period)  
        # temperature difference relative to neighboring cells
        delta_T = T[neighbor_cells] - T[:, None] 
        
        # Heat flux due to conduction
        J = np.sum(k * delta_T / distances * areas, axis=1)
            
        # External heat flux
        J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(surface_normals, r_sun), 0)) - sigma*eps*(T[surface_cells])**4)
        
        # Temeprature change
        dTdt = J/(rho * volumes * cp)
        
        # Updated temperature
        T += dTdt * time_step
    
        # Total thermal force
        F=2/3*eps*sigma/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
        
        # Transfersal thermal force (Yarkovsky force)
        B = np.dot(F, r_trans)
    
        # Semi-major axis drift
        dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
        
        drift.append(dadt)
        
        if np.abs(np.nanmax(drift) - np.nanmin(drift)) > 1e2: # divergence due to the large time step
            
            # Divergence happened. Time step is decreased and the entire calculation starts again.
            total_time = 0
            i = 0
            time_step = time_step / 2
            T = np.zeros(len(volumes)) + T_equilibrium
            drift = []
            
        else:
            
            total_time += time_step
            i += 1

        # Checking for convergence
        if initialization == 1 and np.mod(i, number_of_steps_per_rotation ) == 0 and i >= 2*number_of_steps_per_rotation: # Full rotation completed (starting from the second rotation)
            
            # number of full rotations
            number_of_rotations = i / number_of_steps_per_rotation 
            
            drift_1 = drift[-int(2*number_of_steps_per_rotation):-int(number_of_steps_per_rotation)] # Drift values from the rotation preceding the last one
            drift_2 = drift[-int(number_of_steps_per_rotation):] # Drift values from the last rotation
            
            # Extreme values during the last two rotations
            max_value_1 = np.max(drift_1)
            min_value_1 = np.min(drift_1)
            max_value_2 = np.max(drift_2)
            min_value_2 = np.min(drift_2)
            
            # Amplitude during the final rotation
            amplitude = (max_value_2 - min_value_2)/ np.mean(drift_2)
            
            # Difference between the maxima during the last two rotations
            diff_max = np.abs((max_value_2 - max_value_1)/max_value_2) 
            
            # Difference between the minima during the last two rotations
            diff_min = np.abs((min_value_2 - min_value_1)/min_value_2)
            
            # Difference between the means during the last two rotations
            diff_mean = np.abs((np.mean(drift_2) - np.mean(drift_1))/np.mean(drift_1))
                
            if diff_max < max_tol and diff_min < min_tol and diff_mean < mean_tol and amplitude < amplitude_tol and sphere == 1: # for a spherical asteroid
                
                initialization = 0
                total_time = 0
                i = 0
                T_equilibrium = T.copy()
    
            elif diff_max < max_tol and diff_min < min_tol and diff_mean < mean_tol and sphere == 0: # for a ellipsoidal asteroid

                initialization = 0
                total_time = 0
                i = 0
                T_equilibrium = T.copy()
                
            elif number_of_rotations >= maximum_number_of_rotations: # to prevent too long simulation

                initialization = 0
                total_time = 0
                i = 0
                T_equilibrium = T.copy()
                
    # =============================================================================
    #                       General Yarkovsky         
    # =============================================================================
    print("General Yarkovsky computing started")
    drift = [0, 0]
    
    # total time
    total_time = 0.
    
    # iteration counter
    i=0
    
    time_1 = time.time() # for measuring of the execution time
    total_number_of_iterations = orbital_period / time_step
    total_drift = 0
    while total_time <= orbital_period:
        
        drift[0] = drift[1]
        # position of the Sun in asteroid-fixed reference frame
        r_sun, r_trans, sun_distance, solar_irradiance = sun_position(axis_lat = axis_lat, axis_long = axis_long, 
                                                                      period = rotation_period, 
                                                                      a = semi_major_axis, 
                                                                      e = eccentricity, 
                                                                      t = total_time, 
                                                                      M0 = np.pi, 
                                                                      no_motion = 0, # not needed for isolated seasonal effect
                                                                      precession_period = precession_period)  
        # temperature difference relative to neighboring cells
        delta_T = T[neighbor_cells] - T[:, None] 
        
        # Heat flux due to conduction
        J = np.sum(k * delta_T / distances * areas, axis=1)
            
        # External heat flux
        J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(surface_normals, r_sun), 0)) - sigma*eps*(T[surface_cells])**4)
        
        # Temeprature change
        dTdt = J/(rho * volumes * cp)
        
        # Updated temperature
        T += dTdt * time_step
    
        # Total thermal force
        F=2/3*eps*sigma/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
        
        # Transfersal thermal force (Yarkovsky force)
        B = np.dot(F, r_trans)
    
        # Semi-major axis drift
        dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
        
        drift[1] = dadt
        
        if np.abs(np.nanmax(drift) - np.nanmin(drift)) > 1e2: # divergence due to the large time step
            
            # Divergence happened. Time step is decreased and the entire calculation starts again.
            total_time = 0
            i = 0
            time_step = time_step / 2
            T = T_equilibrium
            drift = [0, 0]
            total_number_of_iterations = orbital_period / time_step
            total_drift = 0
            time_1 = time.time() # for measuring of the execution time
            
        else:
            
            total_time += time_step
            i += 1
            total_drift += dadt
            
        if np.mod(i, 10000)==0 and i > 0:
            
            per_iteration = (time.time() - time_1)/i
            estimated_time = (total_number_of_iterations - i) * per_iteration
            hours, remainder = divmod(int(round(estimated_time)), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = '{:02}:{:02}:{:02}'.format(hours, minutes, seconds)
            print('Estimated execution time is {}'.format(formatted_time))

    return total_drift/total_number_of_iterations

        

    
    return drift