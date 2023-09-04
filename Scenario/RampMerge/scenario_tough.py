#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:08:33 2023

@author: oscar
"""
import random
import numpy as np
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Flow, Mission, Route, Scenario, Traffic, TrafficActor
from smarts.sstudio.types import Distribution, LaneChangingModel, JunctionModel

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Flow, Mission, Route, Scenario, Traffic, TrafficActor

normal = TrafficActor(
    name="car",
    max_speed = 20,
    depart_speed = 'max',
    decel = 2.0,
    emergency_decel = 2.0,
    lane_changing_model = LaneChangingModel(impatience=1, cooperative=0.25,
                                            keep_right=100.0, pushy=1.0),
    junction_model = JunctionModel(ignore_foe_prob=1.0, impatience=1.0),
    min_gap=Distribution(mean=10, sigma=5.0),
    speed = Distribution(mean=random.uniform(0.5, 1.0), sigma=0.0),
)

# flow_name = (start_lane, end_lane)
route_opt = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (2, 2),
    (2, 1),
]

# Traffic combinations = 3C2 + 3C3 = 3 + 1 = 4
# Repeated traffic combinations = 4 * 100 = 400
min_flows = 3
max_flows = 3
route_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
] * 200

traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=("gneE3", start_lane, 0),
                    end=("gneE4", end_lane, "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate= 1800, #* random.uniform(9, 12),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 5),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                actors={normal: 1.0},
                randomly_spaced=True,
            )
            for start_lane, end_lane in routes
        ]
    )

route = Route(begin=("gneE6", 0, 10), end=("gneE4", 2, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=15,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
