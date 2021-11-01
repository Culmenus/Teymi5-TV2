# UV4
Elías Bjartur Einarsson (ebe19), 1. nóvember 2021

## How is a policy gradient method different from the action value methods we have discussed so far?

In the policy gradient method we define a policy function without a value function and update it with regards to its own gradient. In the value and action-value methods we had a value function approximator and the policy was implicit with respects to that, epsilon greedy with respects to the values for example. The policy gradient method has no representation of the value of a state or state-action pair but simply an explicit policy. Of course policy methods can be combined with action-value methods and then there would be a representation of state or state-action values. Policy gradient methods can be beneficial for certain problems, for instance when calculating what is good to do is easier than estimating the true value of your state and possible states resulting from your actions. 

## Describe in your own words how REINFORCE (page 328) solves Exercise 13.1 (see also python code included with the book).

The main reason REINFORCE solves this exercise is because it does not base its policy on a value function that would be bound to be identical for these states under the relevant features. A policy working to maximise the value would be stuck with the same actions for each square while a policy function can choose a probability distribution for the possible actions.
REINFORCE solves this by sampling whole episodes and updating the parameters of the policy function w.r.t. the return and the gradient of (the log of) the policy function. Eventually it will learn to take actions right or left with the optimal probability.  

## What are actor-critic methods and how is it different from REINFORCE with baseline?

While REINFORCE uses a MC approach, the actor-critic methods use a TD approach. That is, REINFORCE first produces a whole episode and uses the full return in its target along with a state-value function for its baseline. The actor-critic methods use for instance the one-step return or the lambda return in its target (alongside the critic's value function). This makes for a fully online and incremental algorithm, something REINFORCE is not.

