# Team Project: Toward Sustainable Cities through simulation
# testing RL

# SUMO imports:
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from typing import Callable, Optional, Tuple, Union

import platform
# on the server (platform = Linux) we use libsumo and also don't need the tools in the path
if platform.system() != "Linux":
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)  # we need to import python modules from the $SUMO_HOME/tools directory
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    import traci
else:
    import libsumo as traci

from sumolib import checkBinary


import numpy as np
import gym
from gym import spaces
from traffic_signal import TrafficSignal

class TLSEnv(gym.Env):
    """
    Environment for the TLS of a simple junction
    """

    MIN_GAP = 2.5  # minimum distance between vehicles
    # The program in the background looks like this:
    #    <tlLogic id="0" type="static" programID="0" offset="0">
    # the locations of the tls are      NESW
    #        <phase duration="100" state="GrGr"/>      <-----------NS
    #        <phase duration="6"  state="yryr"/>
    #        <phase duration="100" state="rGrG"/>      <-----------EW
    #        <phase duration="6"  state="ryry"/>
    #    </tlLogic>

    def __init__(self,
                 netfile: str,
                 cmd: bool = False,
                 reward_fn: Union[str,Callable,dict] = 'diff-waiting-time',
                 observation_fn: Union[str,Callable] = 'default',
                 yellow_time: int = 2):
        super(TLSEnv, self).__init__()
        if cmd:
            self.sumoBinary = checkBinary("sumo")
        else:
            self.sumoBinary = checkBinary('sumo-gui')
        self.sumo = None
        self.stepcnt = 0
        self.change = 0
        self.current = 0
        self.delta_time = 9
        self.run = 0  # whether traci is active
        self._net = netfile
        self.id = "0"  # tls id
        self.reward_fn = reward_fn
        self.observation_fn = observation_fn
        self.last_measure = 0.0
        self.vehicles = dict()
        self.yellow_time = yellow_time
        self.min_green = 5
        self.max_green = 50
        self.begin_time = 0

        self.sumo = traci.start([self.sumoBinary, "-c", "data/"+self._net+".sumocfg", "--start",
                     "--quit-on-end"])
        # Get TS info
        self.ts_ids = list(traci.trafficlight.getIDList())
        self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn,
                                                      traci) for ts in self.ts_ids}
        traci.close()
        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([ts.num_green_phases for ts_id,ts in self.traffic_signals.items()])
        self.id_mapping = dict()
        self._id_mapping()
        
        self.observation_space = spaces.Dict({ts_id: spaces.Box(low=np.zeros(ts.num_green_phases+1+2*len(ts.lanes), dtype=np.float32), high=np.ones(ts.num_green_phases+1+2*len(ts.lanes), dtype=np.float32)) for ts_id,ts in self.traffic_signals.items()})
        self.observation = {ts_id: spaces.Box(low=np.zeros(ts.num_green_phases+1+2*len(ts.lanes), dtype=np.float32), high=np.ones(ts.num_green_phases+1+2*len(ts.lanes), dtype=np.float32)) for ts_id,ts in self.traffic_signals.items()}
        
        self.rewards = {ts: 0 for ts in self.ts_ids}
    def _id_mapping(self):
        id_mapping = {ts_id: 0 for ts_id in self.traffic_signals.items()}
        cnt=0
        for i,_ in id_mapping:
            self.id_mapping.update({i:cnt})
            cnt += 1
    def enable_gui(self):
        self.sumoBinary = checkBinary('sumo-gui')

    def disable_gui(self):
        self.sumoBinary = checkBinary('sumo')

    def reset(self):
        """
        resets the environment

        Important: the observation must be a numpy array
        :return: (np.array)
        """
        if self.run != 0:
            self.close()
        self.run += 1
        self.sumo = traci.start([self.sumoBinary, "-c", "data/"+self._net+".sumocfg", "--start",
                     "--quit-on-end"])
        print("reset start traci    ")
        # traci.load(['-c', 'data/cross.sumocfg', "--start", "--quit-on-end"])
        self.stepcnt = 0
        self.change = 0
        self.current = 0
        self.vehicles = dict()
        self.ts_ids = list(traci.trafficlight.getIDList())
        self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn,
                                                      traci) for ts in self.ts_ids}
        return self._compute_observations()
    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()   
    def step(self, action):
        print("stepping...")
        # if action is None or action.any() is None:
        #     for _ in range(self.delta_time):
        #         traci.simulationStep()
                
        # else:
        print(action)
        self._apply_actions(action)
        self._run_steps()
            
        done = bool(self.stepcnt >= 10001)
        if done:
            print("Episode completed.")
            # traci.close(False)

        # negative sum of accumulated waiting times
        # reward = - (self._get_acc_time("4i_0")+ self._get_acc_time("1i_0") + self._get_acc_time("2i_0"))
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        rewards = self._compute_rewards()
        acc_reward = self._accumulate_rewards(rewards)
        obs = self._compute_observations()
        print(f"reward={acc_reward}")
        return obs, acc_reward, done, info
    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        i=0
        for action in actions:
            ts_id = list(self.id_mapping.keys())[list(self.id_mapping.values()).index(i)]
            i += 1
            if self.traffic_signals[ts_id].time_to_act:
                self.traffic_signals[ts_id].set_next_phase(action)
        print("apply_actions")
    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _sumo_step(self):
        traci.simulationStep()
        self.stepcnt = self.stepcnt+1

    def _compute_observations(self):
        self.observation.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observation[ts].copy() for ts in self.observation if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}
    def _accumulate_rewards(self,rewards):
        sum =0
        
        for k,v in rewards.items():
            sum +=v
        return sum
    def _get_acc_time(self, laneid):
        acc_wt = 0  # accumulated waiting time (sum of all vehicles)
        for veh_id in traci.lane.getLastStepVehicleIDs(laneid):
            acc_wt += traci.vehicle.getAccumulatedWaitingTime(veh_id)
        return acc_wt

    def _pressure_reward(self):
        return -self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def get_accumulated_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh) # lane of vehicle
                acc = traci.vehicle.getAccumulatedWaitingTime(veh) # acc waiting time of vehicle
                if veh not in self.vehicles:  # if vehicle is new on relevant lanes
                    self.vehicles[veh] = {veh_lane: acc}  #add acc waiting time
                else:                        # if vehicle is already on relevant lanes
                    self.vehicles[veh][veh_lane] = acc - sum([self.vehicles[veh][lane] for lane in self.vehicles[veh].keys() if lane != veh_lane]) # and changed lane, then reduce the acc waiting time from the previous lane to get the waiting time on current lane
                wait_time += self.vehicles[veh][veh_lane] # waiting time for one lane
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += traci.vehicle.getSpeed(v) / traci.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def get_teleport_punish(self):
        return -traci.simulation.getEndingTeleportNumber()*100

    def get_out_lanes_density(self):
        lanes_density = [traci.lane.getLastStepVehicleNumber(lane) / (self.lanes_lenght[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane))) for lane in self.out_lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self):
        lanes_density = [traci.lane.getLastStepVehicleNumber(lane) / (self.lanes_lenght[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane))) for lane in self.lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self):
        lanes_queue = [traci.lane.getLastStepHaltingNumber(lane) / (self.lanes_lenght[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane))) for lane in self.lanes]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self):
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_inductionloop_per_lane(self):
        induction = dict()
        ilID = traci.inductionloop.getIDList()
        for ed in ilID:
            induction[traci.inductionloop.getLaneID(ed)] = traci.inductionloop.getLastStepVehicleNumber(ed)
        return [induction[lane] for lane in self.lanes]

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        if self.sumo is None:
            return
        traci.close()
        self.sumo = None
    
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = TLSEnv("data/small.net.xml", cmd=True)
    # If the environment doesn't follow the interface, an error will be thrown
    check_env(env, warn=True)
    env.close()
