## q1 e q2
def lp2lc(coords) -> np.array:
    rho, psi = coords
    return np.array([
        [rho*np.cos(psi)],
        [rho*np.sin(psi)],
    ])

def Q_lp2lc(obs) -> np.array:
    rho, psi = obs
    return np.array([
        [rho**2 * spsi * np.sin(psi)**2 + srho * np.cos(psi)**2, 0.5*(-rho**2 * spsi + srho)*np.sin(2*psi)],
        [0.5*(-rho**2 * spsi + srho)*np.sin(2*psi), rho**2 * spsi * np.cos(psi)**2 + srho * np.sin(psi)**2]
    ])

def R_lc2g(theta: float) -> np.array:
    '''
    Rotation matrix from Local Cartesiano to Global
    '''
    return np.array([
            [ np.cos(theta),-np.sin(theta)],
            [ np.sin(theta), np.cos(theta)]
    ])

def A_lc2g(robot: np.array) -> np.array:
    '''
    Homogenous matrix from Local Cartesiano to Global
    '''    
    x, y, theta = robot
    return np.array([
        [ np.cos(theta), -np.sin(theta), x],
        [ np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def lc2g(coords_lc: np.array, robot: np.array) -> np.array:
    x_lc, y_lc = coords_lc
    return (A_lc2g(robot) @ np.array([[x_lc[0]], [y_lc[0]], [1]]))[:2]

def Q_lc2g(theta: float, Q_lc: np.array) -> np.array:
    return R_lc2g(theta) @ Q_lc @ R_lc2g(theta).T

#q3 e q4 

def g2lc(coords_g: np.array, robot: np.array) -> np.array:
    A_g2lc = LA.inv(A_lc2g(robot))
    gx, gy = coords_g
    return (A_g2lc @ np.array([[gx], [gy], [1]]))[:2]

def C(theta):
    return R_lc2g(theta).T

def lc2lp(coords_lc: np.array) -> np.array:
    l, r = coords_lc
    return np.array([
        [np.sqrt(l**2+r**2)],
        [np.arctan2(r,l)]
    ])