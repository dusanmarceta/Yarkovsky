import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
from constants import G, au, speed_of_light, sigma, S0


#'''
#constants
#'''
#G = 1.3271244e+20
#au = 149597870700.0

def centri(a, nula=0): # radi
    '''
    racuna koordinate centara intervala unutar nekog opsega. Ovo nam treba za
    racunanuje rastojanje izmedju susednih celija
    
    input: opseg podeljen na intervale (koji dobijamo pozivanjem funkcije log_podela)
    
    output: koordinate centara
    '''
    
    
    x=np.zeros(len(a)-1)
    
    for i in range(1,len(a)):
        x[i-1]=a[i-1]+(a[i]-a[i-1])/2
        
    if nula==1: # kada se racunaju centri po r prva celija treba da se nulira
        x[0]=0
        
    return x


def log_podela(broj): # radi
    '''
    Deli opseg na broj delova po logaritamskoj skali. 
    Koristimo za podelu asteroid au radijalnom pravcu kako bi celije blize povrsini
    bile manje u odnosu na one u unutrasnjosti asteroida.
    input: broj celija
    output: podela na intervalu [0,1]
    
    Mnozimo output sa radijusom da bismo dobili podelu na intervalu [0,r]
    '''
    
    podela=[0]
    skala= 1 / np.log10(1.0 + broj);
    
    for i in range(broj):
        
        granica=np.log10(2.0 + i) * skala
        podela.append(granica)
        
    return np.array(podela)
#------------------------------------------------------------------------------
# surfaces of the layers

def kepler(e, M, accuracy):
    # =============================================================================
    # solves Kepler equation using Newton-Raphson method
    # for elliptic and hyperbolic orbit depanding on eccentricity

    # Input:
    # e - eccentricity
    # M - mean anomaly (radians)
    # accuracy - accuracy for Newton-Raphson method (for example 1e-6)
    #
    # Output:
    # E [radians] - eccentric (hyperbolic) anomaly
    # =============================================================================
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
    # =============================================================================
    # converts eccentric (or hyperbolic) anomaly to true anomaly
    # Input:
    # E [radians] - eccentric (or hyperbolic anomaly)
    # Output:
    # True anomaly [radians]
    # =============================================================================
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
        true anomaly
    """
    
    n=np.sqrt(G/(a*au)**3)

    M=n*t + M0
    
    E = kepler(e, M, 1e-6)

    return (a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E))




def sun_position(axis_lat, axis_long, period, a, e, t, M0 = 0, rotation_0 = 0):
    """
    calculates postion of the Sun in the asteroid-fixed reference frame
    Input:
        axis_lat - latitude of the rotation axis wrt inertial frame
        axis_long - longitude of the rotation axis wrt inertial frame
        period - rotational period of the rotation
        time - time from the reference epoch (when meridian of the asteroid pointed toward x-axis of the inertial reference frame)
    output:
        unit vector toward the Sun in asteroid-fixed reference frame
        solar_irradiance - iradiation from the Sun
    """
    
    # instantenous coordinates of the Sun in asteroid-centred inertial reference frame (xOy is orbital plane, x-axis toward pericenter)
    x, y = sun_motion(a, e, M0, t)

    
    r=np.sqrt(x**2+y**2)
    
    nn = np.array([0,0,1]) # normal to orbital plane
    
    [xt, yt, zt] = np.cross(nn, np.array([x/r, y/r, 0])) # unit vector toward the Sun
        
    
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
    rotation_angle = -(2*np.pi/period * t + rotation_0)
    R3 = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                    [0, 0, 1]]);
    
    [x3, y3, z3] = np.matmul(R3, [x2, y2, z2])
    [x3t, y3t, z3t] = np.matmul(R3, [x2t, y2t, z2t])
    
    # rotation from inertial to asteroid-fixed frame
    R=np.linalg.inv(np.matmul(np.matmul(R3, R2), R1))
    
#    R=np.matmul(np.matmul(R1, R2), R3)
    
    '''
    treba proveriti znakove
    '''
    
    
    [xs, ys, zs] = np.matmul(R, np.array([x, y, 0])) # coordinates of the sun in asteroid_fixed reference frame (au)
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    
    ns=[xs/rs, ys/rs, zs/rs] # unit vector toward the Sun
    solar_irradiance = 1361./rs**2 # total solar irradiance at distance rs

    
    tt = np.matmul(R, np.array([xt, yt, zt]))
    
    return([x3, y3, z3], [x3t, y3t, z3t], rs, solar_irradiance)






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
#    return (neighbor_cells, volumes, areas, distances, np.arange(len(facets)), facets_area[0], surface_normals, points_outer, facets)
#    return (facets, points_outer, facets_area, neighbor_cells, volumes, areas, distances, np.arange(len(facets)), facets_area[0], surface_normals)

def numerical_yarko(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layers, rho, k, albedo, cp, eps, axis_lat, axis_long, rotation_period, time_step, semi_major_axis, eccentricity = 0, t = 0, M0 = 0, rotation_0 = 0,  tol = 0.05, lin = 0, conductivity = 1, dt_crit = 1, crit_factor = 3):

    '''
    constants
    '''
    S0 = 1361.  # Zracenje sa Sunca (W/m^2)
    G = 1.3271244e+20
    au = 149597870700.0
    speed_of_light = 299792458. # speed of light
    sig = 5.6704e-8 # Stefan–Boltzmann constant (W/m^2K)
    
    '''
    Ucitavanje mape
    '''
    neighbor_cells, volumes, areas, distances, surface_cells, surface_areas, surface_normals = prismatic_mesh(semi_axis_a, semi_axis_b, semi_axis_c, facet_size, layers)

    d_min = np.min(distances)
    dt_critical=d_min**2*rho*cp/k
    
    if dt_crit == 1:
        time_step = np.min([time_step, dt_critical/crit_factor])

    mean_motion=np.sqrt(G/(semi_major_axis*au)**3)
    mass = 4/3 * semi_axis_a * semi_axis_b * semi_axis_c * np.pi * rho
    
    detektovano=0
    loc_min=0
    loc_max=0
    
    T_equilibrium = (S0/semi_major_axis**2/4/sig)**0.25
    
    T = np.zeros(len(volumes)) + T_equilibrium # temperature svih celija
    
    drift = []
    i=0
    vreme = t
    while True:
        vreme += time_step
        
#        if np.mod(np.floor(vreme/time_step), 100.)==0:
#            print(vreme)
#            progress_file = 'progres_uticaj_dt.txt'
#            zapis = 'i = {}, dt = {}'.format(i, time_step)    
#            np.savetxt(progress_file, [zapis], fmt='%s')
        
        r_sun, r_trans, sun_distance, solar_irradiance, r_test = sun_position(axis_lat, axis_long, rotation_period, semi_major_axis, eccentricity, vreme, M0, rotation_0)
        
        if conductivity == 1:
        
#            delta_T1 = T[okolina_centralne_indeksi] - T[0]
#            J[0] = np.sum(k*delta_T1/np.array(okolina_centralne_rastojanja)*okolina_centralne_povrsine)
            
            delta_T = T[neighbor_cells] - T[:, None] 
            
            J = np.sum(k * delta_T / distances * areas, axis=1)
    
        # external flux
        J[surface_cells] += surface_areas * ((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(surface_normals, r_sun),0))- sig*eps*(T[surface_cells])**4)
    
    
        dTdt = J/(rho * volumes * cp)
        T += dTdt * time_step
        
        F=2/3*eps*sig/ speed_of_light * np.sum((T[surface_cells][:, None])**4 * surface_normals * surface_areas[:, None], axis=0)
        
        B = np.dot(F, r_trans)
    
        dadt = 2 * semi_major_axis / mean_motion / sun_distance * np.sqrt(1 - eccentricity**2) * B / mass
        drift.append(dadt)
        
        if i>2:
            if abs(drift[i-1])>abs(drift[i-2]) and abs(drift[i-1])>abs(drift[i]):
                loc_max = abs(dadt)
                detektovano +=1
                
#                print(detektovano, i, 'max', loc_max)
#                try:
#                    print(abs(drift[i-2]), abs(drift[i-1]), abs(drift[i]))
#                except:
#                    pass
                
            if abs(drift[i-1])<abs(drift[i-2]) and abs(drift[i-1])<abs(drift[i]):
                loc_min = abs(dadt)
                detektovano +=1
                
#                print(detektovano, i, 'min', loc_min)
#                
#                try:
#                    print(abs(drift[i-2]), abs(drift[i-1]), abs(drift[i]))
#                except:
#                    pass
#        
        if  detektovano>=2 and (loc_max - loc_min)/loc_max < tol: # ovo treba proveriti (desava se da uzme dva maksimuma ili dva minimuma)
            rezultat = (loc_max + loc_min)/2
            
            break 
        
        i += 1
        
    if axis_lat > 0:   
        return (rezultat, drift) 
    else:
        return (-rezultat, drift)





def initial_temperature_field(semi_axis_a, semi_axis_b, semi_axis_c, surface_normals, layers, solar_irradiance, Cp, rho, k, albedo, emissivity, rotation_period):
    
    # pracenje specificnih celija
    lat = np.arcsin(np.transpose(surface_normals)[2]) # latitude celija
    long = np.mod(np.arctan2(np.transpose(surface_normals)[1], np.transpose(surface_normals)[0]), 2*np.pi) 
    
 
    
    
    
    Gamma = np.sqrt(rho * Cp * k) # str. 3 ispod eq. 6
    
    omega = 2 * np.pi/rotation_period # ugaona brzina rotacije
    ls = np.sqrt(k/rho/Cp/omega) # eq. 3
    
    
    Tss = ((1-albedo)*solar_irradiance/emissivity/sigma)**(1/4) # Subsolarna temperatura (eq. 5)
    THETA = Gamma * np.sqrt(omega)/emissivity/sigma / Tss**3 # eq. 6
    Tav= ((1-albedo)*solar_irradiance/emissivity/sigma/4)**(1/4) # ovo je zapravo Tss/sqrt(2) kao sto je definisano na str. 3 (markirano zutom bojom)
    
    
    OK = 0
    
    vartheta = np.pi/2 - lat # kolatituda
    phi = np.pi - long # ovo podrazumeva da je u pocetnom trenutku prvi meridijan okrenut ka Suncu
    
    
    R = (semi_axis_a * semi_axis_b * semi_axis_c)**(1/3)
    
    
    T = np.zeros(int(len(long) * len(layers)))
    while OK == 0:
        try:
            Rp = R/ls # str. 3 ispod eq. 6
    
            lam = THETA/np.sqrt(2)/Rp # str. 4 ispod eq. 21
                
            x = np.sqrt(2) * Rp # str. 5, iznad eq. 26 (markirano zutom bojom)
            
            A =  -(x+2.) - np.exp(x)*((x-2.)*np.cos(x) - x * np.sin(x)) # eq. 26
            B =  -x - np.exp(x) * (x * np.cos(x) + (x - 2.)*np.sin(x)) # eq. 27
            C = A + lam/(1.+lam) * (3.*(x+2.) + np.exp(x) * (3.*(x-2.)*np.cos(x) + x * (x-3.)*np.sin(x))) # eq. 28
            D = B + lam/(1.+lam) * (x*(x+3.) - np.exp(x) * (x*(x-3.)*np.cos(x) - 3. * (x-2.)*np.sin(x))) # eq. 29
               
            mnozilac =  (A + B*1j)/(C + D*1j) # eq. 25. Ovo je delilac u drugom clanu u eq. 24
            
            
            # sferni harmonici koji se pojavljuju u eq. 24. Moguce je da je ovde greska posto se oni razliciti normiraju za razlicite primene.
            Y10 = np.sqrt(3/4/np.pi)*np.cos(vartheta)
            Y11 = np.sqrt(3/8/np.pi)*np.sin(vartheta)*np.exp(1j*phi)
            theta_0 = np.pi/2
            
            b10 =  np.sqrt(np.pi/3)*np.cos(theta_0) # eq. 14
            b11 = -np.sqrt(np.pi/6)*np.sin(theta_0) # eq. 15
            
            
            # Eq. 24 !!! Ovo ima i realni i imaginarni deo
            delta_Tp = 1/np.sqrt(2)/(1+lam)*(b10*Y10 + b11*Y11*mnozilac)
            
            # Povrsinska temperatura prema formulama obelezenim crvenom bojom na str. 3
            T_surface = delta_Tp.real * Tss + Tav
    
            OK = 1
            
        except RuntimeWarning:
            R = R*0.9
        
    # Temepratura po dubini
    
    thickness = np.diff(layers)
    thickness = np.insert(thickness, 0, layers[0])
    
    # dubine slojeva
    centers = np.zeros(len(thickness))
    for i in range(len(centers)):
        centers[i] = np.sum(thickness[:i]) + thickness[i]/2
        
    T[:len(long)] = T_surface
        
    for i in range(1, len(centers)):
        
        indeks_0 = i * len(long)
        indeks_1 = (i + 1) * len(long)
        
        slope = (Tav - T_surface) / sum(thickness)
        
        delta_T = slope * centers[i]
        
        T[indeks_0 : indeks_1] = T_surface + delta_T
    return T
       


