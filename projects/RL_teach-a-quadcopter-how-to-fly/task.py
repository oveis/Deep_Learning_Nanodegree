import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles (phi,theta,psi)
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6   # 6-dimentional pose
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4    # 4-dimensional action space, with one entry for each rotor
        self.init_pose = init_pose
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        
    def get_reward_for_takeoff(self):
        """
        Calculate reward for "Taking off Quadcopter" test case.
        The reward is based on the Euclidean distance. Get additional reward for the z-asix distance.
        """
        dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        max_dist = np.linalg.norm(self.sim.upper_bounds - self.init_pose[:3])
        init_source_target_dist = np.linalg.norm(self.target_pos - self.init_pose[:3])

        # Reward based on the distance between current and target locations.
        if dist > init_source_target_dist:
            reward = -1 * (dist / init_source_target_dist)
        else:
            reward = 1 - (dist / max_dist)

        # Additional reward for the z-axis as the agent needs vertical force to takeoff
        tanh_reward = np.tanh(.5 - (abs(self.sim.pose[2] - self.target_pos[2])))
        reward += tanh_reward
        return reward
    
    
    def get_reward_for_landing(self):
        """
        Calculate reward for "Landing Quadcopter" test case.
        The reward is based on the distance between current and target locations.
        Expecting the z_velocity is getting slower to land safely. Used velocity as discount value.
        Ref: https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
        """
        dist_reward = 1 - (np.tanh(abs(self.sim.pose[:3] - self.target_pos)).sum())**0.4
        vel_discount = (1 - max(np.tanh(self.sim.v[2]), 0.1))**(1/max(np.tanh(abs(self.sim.pose[2] - self.target_pos[2])), 0.1))
        reward = vel_discount * dist_reward
        return reward
    
        
    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        # Test Case: Taking off Quadcopter
        # return self.get_reward_for_takeoff()

        # Test Case: Landing Quadcopter 
        return self.get_reward_for_landing()
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # The `done` sets `True` if:
            #   1. the time limit has been exceeded
            #   2. the quadcopter has travelled outside of the bounds of the simulation.
            #          lower_bounds = [-150, -150, 0]
            #          upper_bounds = [150, 150, 300]
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state