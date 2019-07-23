# Distopia Projects

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

