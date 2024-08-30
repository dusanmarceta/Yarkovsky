import numpy as np

'''
constants
'''
G = 1.3271244e+20
au = 149597870700.0


'''
Conversions
'''
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
        tru anomaly
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
    
    return([x3, y3, z3], [x3t, y3t, z3t], rs, solar_irradiance, [x3, y3, z3])

'''
Special functions
'''

def list_min(lista):
    
    '''
    nalazi minimum iz liste koja u sebi ima druge liste
    '''
    
    d_min = 1e6

    for i in range(len(lista)):
        for j in range(len(lista[i])):
            if lista[i][j] < d_min:
                d_min = lista[i][j]
    return d_min

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


def metrika(r1,r2):
    return np.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)

def mesh(R, Nr, Nphi, Ntheta, Nr_shell = None, lin = 0):
    
    '''
    Daje mapu podele asteroida na celije, kao i karakteristike celija koje su potrebne
    za racun prenosa toplote
    
    input:
        R: radijus asteroida
        Nr: podela po radijusu (broj celija u radijalnom pravcu)
        Nphi: podela po longitudi (broj celija po longitudi na intervalu [0, 2pi))
        Ntheta: podela po latitudi (broj celija po latitudi na intervalu [-pi/2, +pi/2))
    
    output:
        celije: niz adresa svake celije u formatu [a,b,c]. a je adresa duz radijalnog pravca, b duz longitude, c duz latitude
        npr. celija [3,3,0] je cetvrta od centra (zato sto je prva sferna celija u centru), cetvrta po longitudi i prva po latitudi (na juznoj strani ose) 
        -----------------
        okolina_adrese: adrese okolnih celija za svaku celiju. 
        okolina_indeksi: indeksi okolnih celija za svaku celiju.
    '''
    
    if Nr_shell is None:
        Nr_shell = Nr

    # Formiranje nizova celija
    if lin == 1:
        r = np.linspace(0, R, Nr + 1)
    else:    
        r = log_podela(Nr)*R
    
    if Nr_shell != 0: 
        r = np.concatenate(([0], r[-Nr_shell:]))
    
#    r=np.array([0,0,R])
    

    phi = np.linspace(0, 2*np.pi, Nphi+1)
    theta = np.linspace(-np.pi/2, np.pi/2, Ntheta+1)

    
    # Odredjivanje centara elemenata
    phi_centri = centri(phi)
#    r_centri = centri(r, nula=1)
    theta_centri = centri(theta)
    
    # Razlika dva susedna ugla (phi, theta)
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]
    
    # ------ Formiranje liste elemenata ------ #
    
    celije = [] # (adrese svake celije) bez centralne celija
    zapremine = [] # zapremine celija
    koordinate_CM=[[0,0,0]]


    for i in range(1, Nr_shell): # ide od 1 da bi se izbacila centralna celija [0,0,0]
        for j in range(Nphi):
            for k in range(Ntheta):

                celije.append([i, j, k]) 
                zapremine.append((r[i+1]**3-r[i]**3)/3*dphi*
                                 (np.sin(theta[k+1])-np.sin(theta[k])))
                
                '''
                koordinate centara masa svake celije. proveriti ovo!
                '''

                C=3/8*(r[i+1]**4-r[i]**4)/(r[i+1]**3-r[i]**3)/(np.sin(theta[k+1])-np.sin(theta[k]))/dphi
                x_CM=C*(np.sin(phi[j+1])-np.sin(phi[j]))*(dtheta+1/2*(np.sin(2*theta[k+1])-np.sin(2*theta[k])))
                y_CM=C*(np.cos(phi[j])-np.cos(phi[j+1]))*(dtheta+1/2*(np.sin(2*theta[k+1])-np.sin(2*theta[k])))
                z_CM=C*(np.sin(theta[k+1])**2-np.sin(theta[k])**2)*dphi
                
                koordinate_CM.append([x_CM, y_CM, z_CM])

    # ------ Formiranje liste susednih celija------- #
    okolina_adrese = []
    okolina_indeksi = []
    okolina_povrsine = []
    okolina_rastojanja=[]
    fiktivna_celija=[len(celije)+1, 0, 0] # dodaje se na kraj i sluzi samo da bi sve celije imale po 6 okolnih. One koje imaju manje, dodaje im se
    # ova fiktivna celija koja ima zapreminu nula, kao i sve povrsine. Na ovaj nacin ne utice na racun, ali omogucuje laksi rad sa nizovima jer ce biti iste duzine.
#    rastojanja_kontrola=[]
    
    for i in range(0, len(celije)):
   
        if celije[i][0] == Nr-1: # ovo su spoljne celije
            
            if celije[i][2] == 0: # naslanjaju se na juznu stranu ose rotacije pa imaju 4 susedne celije
                
                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu (sledece nema jer je na povrsini)
                c1 = fiktivna_celija
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = fiktivna_celija
                c5 = [celije[i][0], celije[i][1], celije[i][2]+1] # sledeca po latitudi (prethodne nema jer je na juznoj strani ose)
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[-2]**2*dphi*(np.sin(-np.pi/2 + dtheta) + 1) # prema centru
                p1 = 0
                p2 = dtheta*0.5*(r[-1]**2-r[-2]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = 0
                p5 = np.cos(-np.pi/2 + dtheta)*dphi*0.5*(r[-1]**2-r[-2]**2) # naredni po latitudi
                      
#                # rastojanja do susednih celija
#                r0 = r_centri[-1] - r_centri[-2]
#                r1 = np.inf # fiktivno
#                r2 = 2*r_centri[-1]*np.sin(dphi/2)*np.sin(dtheta/2)
#                r3 = r2
#                r4 = np.inf # fiktivno
#                r5 = 2*r_centri[-1]*np.sin(dtheta/2)
#                
    
            elif celije[i][2] == Ntheta-1: # naslanjaju se na severnu stranu ose rotacije pa imaju 4 susedne celije
                
                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu (sledece nema jer je na povrsini)
                c1 = fiktivna_celija
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = [celije[i][0], celije[i][1], celije[i][2]-1] # prethodna po latitudi (sledece nema jer je na severnoj strani ose)
                c5 = fiktivna_celija
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[-2]**2*dphi*(1 - np.sin(np.pi/2 - dtheta)) # prema centru
                p1 = 0
                p2 = dtheta*0.5*(r[-1]**2-r[-2]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = np.cos(np.pi/2 - dtheta)*dphi*0.5*(r[-1]**2-r[-2]**2) # prethodni po latitudi
                p5 = 0
                
#                # rastojanja do susednih celija
#                r0 = r_centri[-1] - r_centri[-2]
#                r1 = np.inf
#                r2 = 2*r_centri[-1]*np.sin(dphi/2)*np.sin(dtheta/2)
#                r3 = r2
#                r4 = 2*r_centri[-1]*np.sin(dtheta/2)
#                r5 = np.inf

            else: # ne naslanjaju se na osu rotacije pa imaju 5 susednih celija (a spoljasnje su)
                   
                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu (sledece nema jer je na povrsini)
                c1 = fiktivna_celija
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = [celije[i][0], celije[i][1], (celije[i][2]-1)] # prethodna po latitudi
                c5 = [celije[i][0], celije[i][1], (celije[i][2]+1)] # sledeca po latitudi
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[-2]**2*dphi*(np.sin(theta[celije[i][2] + 1]) - np.sin(theta[celije[i][2]])) # prema centru
                p1 = 0
                p2 = dtheta*0.5*(r[-1]**2-r[-2]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = np.cos(theta[celije[i][2]])*dphi*0.5*(r[-1]**2-r[-2]**2) # prethodni po latitudi
                p5 = np.cos(theta[celije[i][2] + 1])*dphi*0.5*(r[-1]**2-r[-2]**2) # prethodni po latitudi
                
#                # rastojanja do susednih celija
#                r0 = r_centri[-1] - r_centri[-2]
#                r1 = np.inf
#                r2 = 2*r_centri[-1]*np.sin(dphi/2)*np.cos(theta_centri[celije[i][2]])
#                r3 = r2
#                r4 = 2*r_centri[-1]*np.sin(dtheta/2)
#                r5 = r4
                    
        else: # ovo su unutrasnje celije

            if celije[i][2] == 0: # naslanjaju se na juznu stranu ose pa imaju 5 susednih celija

                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu
                c1 = [celije[i][0]+1, celije[i][1], celije[i][2]] # sledeca u radijalnom pravcu
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = fiktivna_celija
                c5 = [celije[i][0], celije[i][1], celije[i][2]+1] # sledeca po latitudi (prethodne nema jer je na juznoj strani ose)
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[celije[i][0]]**2*dphi*(np.sin(-np.pi/2 + dtheta) + 1) # prema centru
                p1 = r[celije[i][0] + 1]**2*dphi*(np.sin(-np.pi/2 + dtheta) + 1) # od centra
                p2 = dtheta*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = 0
                p5 = np.cos(-np.pi/2 + dtheta)*dphi*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # sledeci po latitudi
                
#                # rastojanja do susednih celija
#                r0 = r_centri[celije[i][0]] - r_centri[celije[i][0] - 1]
#                r1 = r_centri[celije[i][0] + 1] - r_centri[celije[i][0]]
#                r2 = 2*r_centri[celije[i][0]]*np.sin(dphi/2)*np.sin(dtheta/2)
#                r3 = r2
#                r4 = np.inf
#                r5 = 2*r_centri[celije[i][0]]*np.sin(dtheta/2)

            elif celije[i][2] == Ntheta-1: # naslanjaju se na severun stranu ose pa imaju 5 susednih celija
               
                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu
                c1 = [celije[i][0]+1, celije[i][1], celije[i][2]] # sledeca u radijalnom pravcu
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = [celije[i][0], celije[i][1], celije[i][2]-1] # prethodna po latitudi (sledece nema jer je na severnoj strani ose)
                c5 = fiktivna_celija
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[celije[i][0]]**2*dphi*(1 - np.sin(np.pi/2 - dtheta)) # prema centru
                p1 = r[celije[i][0] +1 ]**2*dphi*(1 - np.sin(np.pi/2 - dtheta)) # od centra
                p2 = dtheta*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = np.cos(np.pi/2 - dtheta)*dphi*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # prethodni po latitudi
                p5 = 0
                
#                # rastojanja do susednih celija
#                r0 = r_centri[celije[i][0]] - r_centri[celije[i][0] - 1]
#                r1 = r_centri[celije[i][0] + 1] - r_centri[celije[i][0]]
#                r2 = 2*r_centri[celije[i][0]]*np.sin(dphi/2)*np.sin(dtheta/2)
#                r3 = r2
#                r4 = 2*r_centri[celije[i][0]]*np.sin(dtheta/2)
#                r5 = np.inf
            
            else: # unutrasnje koje nisu na osi pa imaju 6 susednih celija
                
                # Adrese okolnih celija
                c0 = [celije[i][0]-1, celije[i][1], celije[i][2]] # prethodna u radijalnom pravcu
                c1 = [celije[i][0]+1, celije[i][1], celije[i][2]] # sledeca u radijalnom pravcu
                c2 = [celije[i][0], (celije[i][1]-1)%Nphi, celije[i][2]] # prethodna po longitudi
                c3 = [celije[i][0], (celije[i][1]+1)%Nphi, celije[i][2]] # sledeca po longitudi
                c4 = [celije[i][0], celije[i][1], celije[i][2]-1] # prethodna po latitudi 
                c5 = [celije[i][0], celije[i][1], celije[i][2]+1] # sledeca po latitudi
                
                # povrsi preko kojih se granici sa susednim celijama
                p0 = r[celije[i][0]]**2*dphi*(np.sin(theta[celije[i][2] + 1]) - np.sin(theta[celije[i][2]])) # prema centru
                p1 = r[celije[i][0] + 1]**2*dphi*(np.sin(theta[celije[i][2] + 1]) - np.sin(theta[celije[i][2]])) # od centra
                p2 = dtheta*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # prethodni po longitudi
                p3 = p2 # naredni po longitudi
                p4 = np.cos(theta[celije[i][2]])*dphi*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # prethodni po latitudi
                p5 = np.cos(theta[celije[i][2] + 1])*dphi*0.5*(r[celije[i][0] + 1]**2-r[celije[i][0]]**2) # sledeci po latitudi
                
#                # rastojanja do susednih celija
#                r0 = r_centri[celije[i][0]] - r_centri[celije[i][0] - 1]
#                r1 = r_centri[celije[i][0] + 1] - r_centri[celije[i][0]]
#                r2 = 2*r_centri[celije[i][0]]*np.sin(dphi/2)*np.cos(theta_centri[celije[i][2]])
#                r3 = r2
#                r4 = 2*r_centri[celije[i][0]]*np.sin(dtheta/2)
#                r5 = r4

        okolina_adrese.append([c0, c1, c2, c3, c4, c5])
        okolina_povrsine.append([p0, p1, p2, p3, p4, p5])
#        rastojanja_kontrola.append([r0, r1, r2, r3, r4, r5])


    # nuliranje svih koordinata centralne celije. Prethodni algoritam ce dati da 
    #je prethodna celija od [1, 2, 3] da je prethodna po radijalnom pravcu 
    # [0, 2, 3] a to je zapravo centralna celija [0,0,0]
    for i in range(len(celije)):
        if okolina_adrese[i][0][0] == 0:
            okolina_adrese[i][0] = [0, 0, 0]
            
    # formiranje okoline centralne celije
    okolina_centralne_adrese = []
    
    for i in range(Nphi):
        for j in range(Ntheta):
            okolina_centralne_adrese.append([1, i, j])
            
    # dodavanje centralne na pocetak liste celija
    celije.insert(0, [0,0,0])
    
    # formiranje niza u kojem su za svaku celiju dati indeksi okolnih celija, a ne njihove adrese po r, fi, theta
    okolina_indeksi = []
    for i in range(len(okolina_adrese)):
        trenutna_okolina = []
        for j in range(len(okolina_adrese[i])):
            try:
                trenutna_okolina.append(celije.index(okolina_adrese[i][j]))
            except:
                trenutna_okolina.append(len(celije))
        okolina_indeksi.append(trenutna_okolina)
        
        
    okolina_centralne_indeksi=[]
    for i in range(len(okolina_centralne_adrese)):
        okolina_centralne_indeksi.append(celije.index(okolina_centralne_adrese[i]))

    '''
    rastojanja do susednih celija (bez centralne, ona ide posebno)
    '''
    
    okolina_rastojanja=np.zeros([len(celije)-1,6])
    
    for i in range(0,len(okolina_rastojanja)): 
        for j in range(6):            
            if okolina_indeksi[i][j] == len(celije): # fiktivna celija
                okolina_rastojanja[i][j]=np.inf   
            else: 
                okolina_rastojanja[i][j]=metrika(koordinate_CM[i+1], koordinate_CM[okolina_indeksi[i][j]])
    
    '''
    CENTRALNA CELIJA
    '''
    

    # rastojanja do susednih od centralne

    okolina_centralne_rastojanja=np.zeros(len(okolina_centralne_indeksi))
    for i in range(len( okolina_centralne_rastojanja)):
         okolina_centralne_rastojanja[i]=metrika([0,0,0], koordinate_CM[okolina_centralne_indeksi[i]] )
    
    
    

    # povrsine za susede od centralne

    okolina_centralne_povrsine=np.zeros(len(okolina_centralne_indeksi))
    for i in range(len(okolina_centralne_indeksi)):
        okolina_centralne_povrsine[i]=r[1]**2*dphi*(np.sin(theta[okolina_centralne_adrese[i][2] + 1]) - np.sin(theta[okolina_centralne_adrese[i][2]]))
            
    # zapremina centralne celije       
    zapremina_centralne=4/3*r[1]**3*np.pi

    '''
    SPOLJNE CELIJE
    
    OVO NE VALJA. Treba normale da se smeste u tezista povrsina, a ne u sredine po koordinatama fi i teta. Priblizno resenje moze biti da se nadje
    teziste za ravanski trougao ili trapez i da onda to bude aproksimacija tezista za sferni poligon.
    '''  
    redni_broj=np.arange((Nr_shell-1)*Nphi*Ntheta + 1) # redni broj svake celije
    spoljne_celije_indeksi=redni_broj[np.transpose(np.array(celije))[0] == Nr_shell-1] # redni broj spoljnih celija
    spoljne_celije_normale=[] # samo za spoljne celije
    spoljne_celije_povrsine=[]

    for i in range(len(spoljne_celije_indeksi)):
        spoljne_celije_normale.append([np.cos(phi_centri[celije[spoljne_celije_indeksi[i]][1]])*np.cos(theta_centri[celije[spoljne_celije_indeksi[i]][2]]),
                          np.sin(phi_centri[celije[spoljne_celije_indeksi[i]][1]])*np.cos(theta_centri[celije[spoljne_celije_indeksi[i]][2]]),
                          np.sin(theta_centri[celije[spoljne_celije_indeksi[i]][2]])])

        spoljne_celije_povrsine.append(R**2*dphi*(np.sin(theta[celije[spoljne_celije_indeksi[i]][2] + 1]) - 
        np.sin(theta[celije[spoljne_celije_indeksi[i]][2]])))

    zapremine.insert(0, zapremina_centralne)
    return (celije, okolina_indeksi, np.array(zapremine), np.array(okolina_povrsine), np.array(okolina_rastojanja), 
            okolina_centralne_indeksi, okolina_centralne_povrsine, okolina_centralne_rastojanja,
            spoljne_celije_indeksi, np.array(spoljne_celije_povrsine), np.array(spoljne_celije_normale))

                

def numerical_yarko(R, rho, k, albedo, cp, eps, axis_lat, axis_long, period_rotacije, Nr, Nphi, Ntheta, Nr_shell, time_step, a, e = 0, t = 0, M0 = 0, rotation_0 = 0,  tol = 0.05, lin = 0, conductivity = 1, dt_crit = 1, crit_factor = 3):

    '''
    constants
    '''
    S0 = 1361.  # Zracenje sa Sunca (W/m^2)
    G = 1.3271244e+20
    au = 149597870700.0
    c = 299792458. # speed of light
    sig = 5.6704e-8 # Stefan–Boltzmann constant (W/m^2K)
    
    '''
    Ucitavanje mape
    '''
    celije, okolina_indeksi, zapremine, okolina_povrsine, okolina_rastojanja, \
    okolina_centralne_indeksi, okolina_centralne_povrsine, \
    okolina_centralne_rastojanja, spoljne_celije_indeksi, spoljne_celije_povrsine, \
    spoljne_celije_normale=mesh(R, Nr, Nphi, Ntheta, Nr_shell, lin)

    d_min = np.min(okolina_rastojanja)
    dt_critical=d_min**2*rho*cp/k
    
    if dt_crit == 1:
        time_step = np.min([time_step, dt_critical/crit_factor])


    mean_motion=np.sqrt(G/(a*au)**3)
    mass=4/3*R**3*np.pi*rho
    
    detektovano=0
    loc_min=0
    loc_max=0
    
    N = len(celije) + 1 # dodaje se fiktivna celija
    
    T_equilibrium = (S0/a**2/4/sig)**0.25
    
    T = np.zeros(N)+T_equilibrium; # temperature svih celija
    J = np.zeros(N)

    drift = []
    i=0
    vreme = t
    while True:
        vreme += time_step
        
        r_sun, r_trans, sun_distance, solar_irradiance, r_test = sun_position(axis_lat, axis_long, period_rotacije, a, e, vreme, M0, rotation_0)
        
        # Array broadcasting
        J = np.zeros(N)
        
        if conductivity == 1:
        
            delta_T1 = T[okolina_centralne_indeksi] - T[0]
            J[0] = np.sum(k*delta_T1/np.array(okolina_centralne_rastojanja)*okolina_centralne_povrsine)
            
            delta_T1 = T[okolina_indeksi] - T[1:-1, None] 
            
            J[1:-1] = np.sum(k*delta_T1/np.array(okolina_rastojanja)*okolina_povrsine, axis=1)
    
        J[spoljne_celije_indeksi] += spoljne_celije_povrsine*((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(spoljne_celije_normale, r_sun),0))- sig*eps*(T[spoljne_celije_indeksi])**4)
    
    
        dTdt = J[:-1]/(rho*zapremine*cp)
        T[:-1] += dTdt * time_step
        
        F=2/3*eps*sig/c*np.sum((T[spoljne_celije_indeksi][:, None])**4 * spoljne_celije_normale * spoljne_celije_povrsine[:, None], axis=0)
        
        B = np.dot(F, r_trans)
    
        dadt = 2*a/mean_motion/sun_distance * np.sqrt(1-e**2)*B/mass
        drift.append(dadt)
        
        if i>2:
            if abs(drift[i-1])>abs(drift[i-2]) and abs(drift[i-1])>abs(drift[i]):
                loc_max = abs(dadt)
                detektovano +=1

                
            if abs(drift[i-1])<abs(drift[i-2]) and abs(drift[i-1])<abs(drift[i]):
                loc_min = abs(dadt)
                detektovano +=1
                

        if  detektovano>=2 and (loc_max - loc_min)/loc_max < tol: # ovo treba proveriti (desava se da uzme dva maksimuma ili dva minimuma)
            rezultat = (loc_max + loc_min)/2
            
            break 
        
        i += 1
        
    if axis_lat > 0:   
        return rezultat 
    else:
        return -rezultat
    
    

def numerical_yarko_shell(R, rho, k, albedo, cp, eps, axis_lat, axis_long, period_rotacije, Nr, Nphi, Ntheta, Nr_shell, time_step, a, e = 0, t = 0, M0 = 0, rotation_0 = 0,  tolerancija = 0.05, dt_crit = 1, crit_factor = 3):

    '''
    constants
    '''
    S0 = 1361.  # Zracenje sa Sunca (W/m^2)
    G = 1.3271244e+20
    au = 149597870700.0
    c = 299792458. # speed of light
    sig = 5.6704e-8 # Stefan–Boltzmann constant (W/m^2K)
    
    '''
    Ucitavanje mape
    '''
    celije, okolina_indeksi, zapremine, okolina_povrsine, okolina_rastojanja, \
    okolina_centralne_indeksi, okolina_centralne_povrsine, \
    okolina_centralne_rastojanja, spoljne_celije_indeksi, spoljne_celije_povrsine, \
    spoljne_celije_normale=mesh(R, Nr, Nphi, Ntheta)


    d_min = np.min(okolina_rastojanja)
    dt_critical=d_min**2*rho*cp/k
    
    if dt_crit == 1:
        time_step = np.min([time_step, dt_critical/crit_factor])


    mean_motion=np.sqrt(G/(a*au)**3)
    mass=4/3*R**3*np.pi*rho
    
    detektovano=0
    loc_min=0
    loc_max=0
    
    N = len(celije) + 1 # dodaje se fiktivna celija
    
    T_equilibrium = (S0/a**2/4/sig)**0.25
    
    T = np.zeros(N)+T_equilibrium; # temperature svih celija
    J = np.zeros(N)

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
        
        r_sun, r_trans, sun_distance, solar_irradiance, r_test = sun_position(axis_lat, axis_long, period_rotacije, a, e, vreme, M0, rotation_0)
        
        # Array broadcasting
        J = np.zeros(N)
        
        delta_T1 = T[okolina_centralne_indeksi] - T[0]
        J[0] = np.sum(k*delta_T1/np.array(okolina_centralne_rastojanja)*okolina_centralne_povrsine)
        
        delta_T1 = T[okolina_indeksi] - T[1:-1, None] 
        
        J[1:-1] = np.sum(k*delta_T1/np.array(okolina_rastojanja)*okolina_povrsine, axis=1)
    
        J[spoljne_celije_indeksi] += spoljne_celije_povrsine*((np.maximum(solar_irradiance * (1-albedo) * 
         np.dot(spoljne_celije_normale, r_sun),0))- sig*eps*(T[spoljne_celije_indeksi])**4)
    
    
        dTdt = J[:-1]/(rho*zapremine*cp)
        T[:-1] += dTdt * time_step
        
        F=2/3*eps*sig/c*np.sum((T[spoljne_celije_indeksi][:, None])**4 * spoljne_celije_normale * spoljne_celije_povrsine[:, None], axis=0)
        
        B = np.dot(F, r_trans)
        '''
        prvo se sila koja se dobija iz Spitale (eq: 11) razlozi na komponente R (u pravcu radijus vekora) 
        i B (normalna na radijus vektor u ravni orbite) (Danby, 1992 eq 11.5.1 str. 323)
        
        Onda se da/dt dobije iz Danby eq:11.5.13 str. 327
        '''
        dadt = 2*a/mean_motion/sun_distance * np.sqrt(1-e**2)*B/mass
        drift.append(dadt)
        
        if i>2:
            if abs(drift[i-1])>abs(drift[i-2]) and abs(drift[i-1])>abs(drift[i]):
                loc_max = abs(dadt)
                detektovano +=1
            if abs(drift[i-1])<abs(drift[i-2]) and abs(drift[i-1])<abs(drift[i]):
                loc_min = abs(dadt)
                detektovano +=1
        
        if  detektovano>=2 and (loc_max - loc_min)/loc_max < tolerancija: # ovo treba proveriti (desava se da uzme dva maksimuma ili dva minimuma)
            rezultat = (loc_max + loc_min)/2
            break 
        i += 1
        
    if axis_lat > 0:   
        return rezultat
    else:
        return -rezultat
    
'''
LINEAR MODEL
'''
                
               # Constants
pi = np.pi
twopi = 2 * np.pi
deg2rad = pi / 180.0
rad2deg = 180.0 / pi
gmsun = 1.32712440018e20  # gravitational constant * mass of Sun [m^3/s^2]
uGc = 6.67430e-11  # universal gravitational constant [m^3 kg^-1 s^-2]
au2m = 1.495978707e11  # astronomical unit to meters
m2au = 1 / au2m
s2my = 1 / (365.25 * 24 * 3600 * 1e6)  # seconds to million years
h2s = 3600  # hours to seconds
cLight = 299792458.  # speed of light [m/s]
lumSun = 3.828e26  # solar luminosity [W]
sBoltz = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]

def compute_depths(R, rho, K, C, sma, P):
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (sma * au2m)**3.0)
    omega_rot = twopi / (P * h2s)
    ls = np.sqrt(K / (rho * C * omega_rev))
    ld = ls * np.sqrt(omega_rev / omega_rot)
    return ls, ld

def computeYarkoMaxMin_circular(rho, K, C, radius, semiaxm, rotPer, alpha, epsi):
    yarkomax = computeYarko_circular(rho, K, C, radius, semiaxm, 0.0, rotPer, alpha, epsi)
    f90 = computeYarko_circular(rho, K, C, radius, semiaxm, 90.0, rotPer, alpha, epsi)
    f180 = computeYarko_circular(rho, K, C, radius, semiaxm, 180.0, rotPer, alpha, epsi)
    if abs(yarkomax / (2.0 * f90)) < 1.0:
        gam = np.arccos(yarkomax / (2.0 * f90)) * rad2deg
        yarkoadd = computeYarko_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
        yarkomin = min(yarkoadd, f180)
        gammin = gam
    else:
        yarkomin = f180
        gammin = 180.0
    return yarkomax, yarkomin, gammin

def computeYarko_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi):
    das = yarko_seasonal_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
    dad = yarko_diurnal_circular(rho, K, C, radius, semiaxm, gam, rotPer, alpha, epsi)
    yarko = (das + dad) * m2au * s2my
    return yarko

def yarko_seasonal_circular(rho, K, C, R, a0, gam, rotPer, alpha, epsi):
    gam_rad = gam * deg2rad
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (a0 * au2m)**3.0)
    ls = np.sqrt(K / (rho * C * omega_rev))
    Rps = R / ls
    dist = a0 * au2m
    E_star = lumSun / (4.0 * pi * dist**2.0)
    phi = pi * R**2.0 * E_star / (mAst * cLight)
    T_star = (alpha * E_star / (epsi * sBoltz))**0.25
    Theta = np.sqrt(rho * K * C * omega_rev) / (epsi * sBoltz * T_star**3.0)
    F_omega_rev = Fnu_eval(Rps, Theta)
    yarko_s = 4.0 * alpha * phi * F_omega_rev * np.sin(gam_rad)**2.0 / (9.0 * omega_rev)
    return yarko_s

def yarko_diurnal_circular(rho, K, C, R, a0, gam, rotPer, alpha, epsi):
    gam_rad = gam * deg2rad
    mAst = 4.0 * pi * rho * R**3.0 / 3.0
    mu = gmsun + uGc * mAst
    omega_rev = np.sqrt(mu / (a0 * au2m)**3.0)
    omega_rot = twopi / (rotPer * h2s)
    ld = np.sqrt(K / (rho * C * omega_rot))
    Rpd = R / ld
    dist = a0 * au2m
    E_star = lumSun / (4.0 * pi * dist**2.0)
    phi = pi * R**2.0 * E_star / (mAst * cLight)
    T_star = (alpha * E_star / (epsi * sBoltz))**0.25 
    Theta = np.sqrt(rho * K * C * omega_rot) / (epsi * sBoltz * T_star**3.0)
    F_omega_rot = Fnu_eval(Rpd, Theta)
    yarko_d = -8.0 * alpha * phi * F_omega_rot * np.cos(gam_rad) / (9.0 * omega_rev)
    return yarko_d

def Fnu_eval(R, Theta):
    if R > 30.0:
        k1, k2, k3 = 0.5, 0.5, 0.5
    else:
        x = np.sqrt(2.0) * R
        A = -(x + 2.0) - np.exp(x) * ((x - 2.0) * np.cos(x) - x * np.sin(x))
        B = -x - np.exp(x) * (x * np.cos(x) + (x - 2.0) * np.sin(x))
        U = 3.0 * (x + 2.0) + np.exp(x) * (3.0 * (x - 2.0) * np.cos(x) + x * (x - 3.0) * np.sin(x))
        V = x * (x + 3.0) - np.exp(x) * (x * (x - 3.0) * np.cos(x) - 3.0 * (x - 2.0) * np.sin(x))
        den = x * (A**2.0 + B**2.0)
        k1 = (A * V - B * U) / den
        k2 = (A * (A + U) + B * (B + V)) / den
        k3 = ((A + U)**2.0 + (B + V)**2.0) / (den * x)
    Fnu = -k1 * Theta / (1.0 + 2.0 * k2 * Theta + k3 * Theta**2.0)
    return Fnu 