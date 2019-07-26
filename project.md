# Distopia Projects

## Overview
Design tasks tend to have sparse state spaces in ways that can be hard to model. For example, among the millions of combinations of food ingredients, only a few can form the basis of useful recipes, whether due to chemistry, practicality, or taste. Humans are very good at navigating this sparseness, and, in the context of human-AI collaborative design, it may be a waste to train AI agents to do so, instead of relying on the human to keep the search "reasonable". In this work, we propose a method to train AI design agents that only know how to optimize over such reasonable designs. We evaluate how this affects learning speed and convergence, and test the relative performance of agents trained this way in a collaborative setting.


## Reasonable Design States for RL
In design tasks, the state space is typically larger than the set of "reasonable" designs. 

"Unreasonable" designs may not be easy to distinguish based on reward, for example a pizzeria with sushi robots may be quite profitable.

An ideal optimal policy defines how to achieve a good design from either a reasonable or unreasonable design.

However, in AI-human collaborative design, we believe that humans are comparatively good at constraining the search to "reasonable" design states.

If this is the case, then a policy that is not trained on unreasonable states might perform almost as well as a much more expensive-to-train policy that is optimal over all states.

It is also possible that a "reasonable" policy is locally more optimal for "reasonable" search (as maintained by the human).

Our hypothesis is that if we can sample reasonable initial states, agnostic of outcome, then we can learn policies
that enable meaningful design collaboration with humans faster than without.

Concretely, we will use a GAN to generate valid start states (in Distopia). Compare this with an agent trained from a fixed valid start and 
with an agent trained on initial states sampled from the data used to train the GAN. Is the GAN-backed agent more robust?

Note that Distopia provides a toy problem where the ground-truth (valid/invalid) is available to us. 
SKN presents a more interesting problem where the structure of "reasonability" must be inferred.

## Reasonable Design in Distopia
Distopia is a user interface that lets human designers draw voting districts. From a human perspective, Distopia's main challenges involve figuring out what factors play into fair districts and how they interact with each other. When training a collaborative agent, however, we found that the agent spends a lot of time exploring configurations that are trivially unacceptable. For example, it is possible to draw less than 8 districts, or to leave out certain precincts, or to make non-contiguous districts. All of these characteristics invalidate a design, and a human, with feedback from the interface, is able to mostly avoid these pitfalls when searching. However, with random starting states (which is necessary given the large design space), the agent has to learn how to both avoid these states, and how to escape them when it encounters them.

One solution to could be to encode these constraints in the reward. Eventually, the agent would learn not to create designs that violate these constraints. The agent would also ideally learn how to recover from designs that violate these constraints.
However, we argue that, if the agent is only meant to collaborate with the human, and the human is trivially able to avoid and correct these invalid states, then it is a waste of time to learn a policy over this more difficult reward. We should, of course, still teach the agent not to go towards unacceptable states; however, we don't need to teach it how to escape, which could save a lot of training time.

Concerns: Some of this difficulty is the result of the shared interface, as opposed to the problem itself. This is less of a problem in the kitchen domain. Also, maybe there is value in teaching the agent to escape bad states--how do we quantify the potential value this could offer and whether it is worth the extra training time?

## Reasonable Design in SKN
Sophie's Kitchen Nightmares is a restaurant simulation game. Using the same block-based interface as Distopia, SKN designers lay out kitchen appliances and furniture to construct a restaurant, balancing outcomes like profit and customer ratings. Unreasonable designs in SKN include those, in a similar vein to Distopia, without any cooking equipment or seating. However, there are also other types of "unreasonable-ness". For example, certain types of furniture or appliances may not go well together. A brick-fired pizza oven may be out of place in an ice cream shop, for example. Learning how to improve these types of extremely weird configurations could also waste training time.

Of course, unlike invalid voting districts, it is harder to formalize these constraints, so sampling "reasonable" restaurants becomes even harder. Simply collecting user-designed restaurants to sample from may work, or it may not provide enough variety. Nor can we use agent exploration data to augment human-collected data for our sampling, as we can do in Distopia, because we don't have a clear ground truth for "unreasonable". Thus, in the SKN case, training a generator and discriminator together becomes much more useful.

Note also that we are not trying to solve the reframing problem for the agent. That is, we are not trying to help the agent avoid weird or unreasonable designs in its own search. Again, we are simply trying to avoid training the agent to handle weird situations it is unlikely to encounter in real life, and we use human-generated data, augmented through a GAN, to approximate those boundaries.

Concerns:
Maybe the set of "reasonable" designs is not a small subset of all restaurant designs. I suppose the more genres of food that we add to the simulator, the smaller the subset gets. But one concern is that there is very little time savings in ignoring those cases. Also, in Distopia, because illegal district assignments lead to low rewards, it is easy to evaluate a change in the learning rate. But it may not be the case with SKN; I think if there is a beneficial effect, it will be because we reduce the variance in our training data and are "overfitting" to the situations we give it. We can argue that this is more efficient, but it also arguably reduces the ability of the agent to generalize between domains, to find underlying themes, and to make "innovative" suggestions. On the other hand, it could also improve the agent's ability to make these kinds of suggestions, by reducing what is essentially, for us, noise. Understanding how to measure these sorts of questions will be critical.


## Additional Questions
If we have a measure of "reasonableness" or "appropriateness", especially one driven by human-generated data, should we also use this to augment the reward? This recalls prior work on reward-shaping via preference learning. Note that a big difference between our approaches is that cherry-picking start states optimizes limited training time, reducing the policy space by ignoring policies that account for invalid start states. However, it does nothing to encourage the agent to avoid invalid or unreasonable states. Preference learning (and our future work on problem framing) does the latter. However, we can possibly support preference learning by providing access to a latent-space model of reasonable designs, e.g. by adding the discriminator to the reward function.


## GAN stuff
* use sigmoid instead of tanh activation in generator (since values between 0 and 1)
* move to Pytorch?
* logging etc.

## Agent Experiments
* train w/ random restarts from valid samples
* punishments? e.g. -1 for invalid move
* build visualization pipeline
* build up test suite
* revamp checkpointing so we don't lose weights
