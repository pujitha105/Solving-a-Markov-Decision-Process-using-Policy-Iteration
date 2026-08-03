# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    while True:

        delta = 0

        for s in range(n_states):

            v = V[s]
            new_v = 0

            # Bellman Expectation Equation
            for a, action_prob in enumerate(policy[s]):

                for prob, next_state, reward, done in env.P[s][a]:

                    new_v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        if delta < theta:
            break

    return V



# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------

def policy_improvement(env, V, gamma=0.99):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.zeros((n_states, n_actions))

    for s in range(n_states):

        action_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob, next_state, reward, done in env.P[s][a]:

                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0

    return policy

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initial Random Policy
    policy = np.ones((n_states, n_actions)) / n_actions

    # Save initial policy
    initial_policy = policy.copy()

    # Evaluate initial policy
    initial_value_function = policy_evaluation(
        env,
        initial_policy,
        gamma,
        theta
    )

    iterations = 0

    while True:

        # Policy Evaluation
        V = policy_evaluation(env, policy, gamma, theta)

        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma)

        iterations += 1

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return (
        policy,
        V,
        iterations,
        initial_policy,
        initial_value_function
    )


```

## Output



<img width="575" height="502" alt="image" src="https://github.com/user-attachments/assets/3cedc9d6-2972-42c8-ab76-3b602183f96e" />







---

## Result


The Policy Iteration algorithm was successfully implemented on the Gymnasium FrozenLake-v1 environment. The algorithm iteratively performed policy evaluation and policy improvement until the policy became stable. It generated the initial and optimal state-value functions along with the corresponding policies, demonstrating convergence to the optimal policy for the given Markov Decision Process.

---

## Inference


From this experiment, it is inferred that the Policy Iteration algorithm efficiently solves a finite Markov Decision Process by repeatedly evaluating and improving the policy. The algorithm converges to the optimal policy in a finite number of iterations, producing the optimal state-value function and the best action for every state. This demonstrates the effectiveness of Dynamic Programming techniques in reinforcement learning for solving known environments such as FrozenLake.

