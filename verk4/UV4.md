# UV4
Elías Bjartur Einarsson (ebe19), 1. nóvember 2021

## How is a policy gradient method different from the action value methods we have discussed so far?
Policy gradient relies on optimizing parametrized policies with respect to the expected return (long-term cumulative reward) by gradient descent.

The core difference between policy gradient methods and previous, purely value-based methods is that the policy is parametrized rather than implied from the value function. This allows for the policy to be updated independent of any estimate of the value function, although a value function may still be used in conjunction with these methods.

In the value and action-value methods we had a value function approximator and the policy was implicit with respect to that, epsilon greedy with respect to the values for example. The policy gradient method has no representation of the value of a state or state-action pair but simply an explicit policy. Of course policy methods can be combined with action-value methods and then there would be a representation of state or state-action values.

Policy gradient methods can be beneficial for certain problems, for instance when calculating what is good to do is easier than estimating the true value of your state and possible states resulting from your actions. As it doesn’t suffer from the problems that the classic reinforcement learning is having such as such as the lack of guarantees of a value function, the intractability problem resulting from uncertain state information and the complexity arising from continuous states & actions.

## Describe in your own words how REINFORCE (page 328) solves Exercise 13.1 (see also python code included with the book).

The main reason REINFORCE solves this exercise is because it does not base its policy on a value function that would be bound to be identical for these states under the relevant features. A policy working to maximise the value would be stuck with the same actions for each square while a policy function can choose a probability distribution for the possible actions.
REINFORCE solves this by sampling whole episodes and updating the parameters of the policy function w.r.t. the return and the gradient of (the log of) the policy function. Eventually it will learn to take actions right or left with the optimal probability.  

## What are actor-critic methods and how is it different from REINFORCE with baseline?

While REINFORCE uses a MC approach, the actor-critic methods use a TD approach. That is, REINFORCE first produces a whole episode and uses the full return in its target along with a state-value function for its baseline. The actor-critic methods use for instance the one-step return or the lambda return in its target (alongside the critic's value function). This makes for a fully online and incremental algorithm, something REINFORCE is not.
