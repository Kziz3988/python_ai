import numpy as np
from typing import Dict, List, Tuple
__all__ = ["PType", "policy_iteration", "QLearning"]
PType = Dict[
    int,
    Dict[
        int,
        List[
            Tuple[
                float,
                int,
                int,
                bool
            ]
        ]
    ]
]

def policy_evaluation(
    P: PType,
    nS: int,
    nA: int,
    policy: np.ndarray,
    gamma: float = 0.9,
    tol: float = 1e-3
)-> np.ndarray:
    value_function = np.zeros(nS)
    while True:
        delta = 0.0
        for s in range(nS):
            a = policy[s]
            v = 0
            for p in P[s][a]:
                prob, _s, r, end = p
                v += (prob * (r + gamma * value_function[_s] * (1 - int(end))))
            delta = max(delta, abs(v - value_function[s]))
            value_function[s] = v
        if delta < tol:
            break
    return value_function

def policy_improvement(
P: PType,
nS: int,
nA: int,
value_from_policy: np.ndarray,
policy: np.ndarray,
gamma: float = 0.9
)-> np.ndarray:
    new_policy = np.zeros(nS, dtype="int")
    for s in range(nS):
        Q = np.zeros(nA)
        for a in range(nA):
            for p in P[s][a]:
                prob, _s, r, end = p
                Q[a] += (prob * (r + gamma * value_from_policy[_s] * (1 - int(end))))
        new_policy[s] = np.argmax(Q)
    return new_policy

def policy_iteration(
    P: PType,
    nS: int,
    nA: int,
    gamma: float = 0.9,
    tol: float = 1e-3
)-> Tuple[np.ndarray, np.ndarray]:
    value_function = np.zeros(nS)
    policy = np.random.randint(0, 4, size = nS)
    while True:
        policy_ = policy.copy()
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        if np.array_equal(policy, policy_):
            break
    return value_function, policy

import gymnasium
def QLearning(
    env:gymnasium.Env,
    num_episodes=2000,
    gamma=0.9,
    lr=0.1,
    epsilon=0.8,
    epsilon_decay=0.99
)-> np.ndarray:
    nS:int = env.observation_space.n
    nA:int = env.action_space.n
    Q = np.zeros((nS, nA))
    max_steps = 100
    for i in range(num_episodes):
        s = env.reset()[0]
        end = False
        stop = False
        steps = 0
        while not end and not stop and steps < max_steps:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s, :])
            _s, r, end, stop, _ = env.step(a)
            Q[s, a] += (lr * (r + gamma * np.max(Q[_s, :]) - Q[s, a]))
            s = _s
            steps += 1
        epsilon = epsilon * epsilon_decay
    return Q
