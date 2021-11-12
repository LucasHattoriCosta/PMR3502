import numpy as np
import numpy.linalg as LA

srho = 0.25**2
spsi = 0.0436**2
alpha = 0.05

class EKF_SLAM:
    def __init__(self, landmarks, observations, robot_t0):
        self.landmarks = landmarks
        self.observations = observations
        self.robot = robot_t0
        
    @staticmethod
    def Lj(land, obs, robot_t):
        '''
        land = [mu, sigma]
        mu = [x_bar, y_bar]
        obs = [rho, psi]
        '''
        return C(robot_t[2]) @ land['sigma'] @ C(robot_t[2]).T + Q_lp2lc(obs)
    
    def W(self, observations_t, robot_t):
        '''
        Observations_t = todas as obs de um mesmo timestamp
        Robot_t = estado do robo no timestamp
        '''
        m = len(self.landmarks)
        n = len(observations_t)
        W = np.empty((n,m))
        for i, obs in enumerate(observations_t):
            for j, land in enumerate(self.landmarks):
                d = lp2lc(obs) - g2lc(land['mu'], robot_t)
                L = self.Lj(land, obs, robot_t)
                W[i][j] = 1 / np.sqrt(LA.det(2*np.pi*L)) * np.exp(-(d.T @ LA.inv(L) @ d)/2)
        
        self.W = W
        return W

    def attribute_landmarks(self, observations_t, robot_t):
        W = self.W.copy()
        c = []
        new_landmarks = []
        n, m = W.shape
        while True:
            i,j = np.unravel_index(np.argmax(W), W.shape)
            max_W = np.max(W)
            if max_W == -1:
                break
            elif max_W >= alpha:
                c.append({'observation':i, 'landmark': j, 'weight': max_W})
                W[i,:] = -1
                W[:,j] = -1
            else:
                #create new landmark
                obs = observations_t[i]
                initial_mu = lc2g(lp2lc(obs), robot_t)
                initial_sigma = Q_lc2g(robot_t[2], Q_lp2lc(obs))
                new_landmarks.append({'mu':initial_mu, 'sigma':initial_sigma})
                c.append({'observation':i, 'landmark': m+len(new_landmarks)-1, 'weight': alpha})
                W[i,:] = -1
        
        self.attributions = c
        self.new_landmarks = new_landmarks
        return c, new_landmarks
    
    def Kalman_Gain(self,land, robot_t, obs):
        K = land['sigma'] @ C(robot_t[2]).T @ LA.inv(self.Lj(land, obs, robot_t))
        return K
    
    def update_map(self, observation_t, robot_t):
        for i, att in enumerate(self.attributions):
            if att['landmark'] >= len(self.landmarks):
                pass
            else:
                land = self.landmarks[att['landmark']]
                obs = observation_t[att['observation']]
                d = lp2lc(obs) - g2lc(land['mu'], robot_t)
                K = self.Kalman_Gain(land,robot_t,obs)
                land['mu'] += (K @ d).T[0]
                land['sigma'] = (np.eye(K.shape[0]) - K @ C(robot_t[2])) @ land['sigma']
                self.landmarks[att['landmark']] = land
                
    def ekf(self):
        for observation_t in self.observations:
            self.W(observation_t, self.robot)
            self.attribute_landmarks(observation_t, self.robot)
            self.correction(observation_t, self.robot)
            return self.attributions, self.landmarks

