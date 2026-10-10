something more fundamental

Yes. The fundamental payoff is not “we found a gravity direction.” It is determining **what kind of representation V-JEPA has learned**.

Current work already shows that physical information emerges across layers and can have structured, distributed geometry. In *Interpreting Physics in Video World Models*, motion-related information becomes accessible in a “Physics Emergence Zone,” and direction is represented as a high-dimensional population code rather than a neat classical-physics coordinate. [arXiv](https://arxiv.org/abs/2602.07050?utm_source=chatgpt.com) That still leaves a deeper question unanswered:

\$$ \boxed{ \text{Is V-JEPA's latent space organized by the \emph{causes} generating a world, or only by the observable trajectories those causes produce?} } \$$

That distinction is fundamental because a representation of **effects** can interpolate familiar motion, while a representation of **mechanisms** could support modular counterfactuals, recombination, and transfer when the visual consequences change.

## A stronger thesis: is the latent a coordinate system for physical mechanisms?

Let the simulator generate a video as

\$$ X = F(\theta,s_0,\eta) \$$

where

\$$ \theta=(g,e,\mu) \$$

contains physical mechanisms,

\$$ s_0=(x_0,v_0,\ldots) \$$

contains initial state, and

\$$ \eta=(\text{camera, colour, lighting, shape}) \$$

contains nuisance variables.

At V-JEPA layer \$l\$,

\$$ z_l=E_l(X). \$$

So jointly,

\$$ z_l=\Phi_l(\theta,s_0,\eta). \$$

The deep question is:

> **Has \$\Phi_l\$ reorganized the observations into coordinates reflecting the generating mechanisms?**

That can be tested using the **local geometry** of the representation rather than another probe.

### The key object: the causal Jacobian

Define

\$$ \boxed{ J_\theta^{(l)} = \frac{\partial z_l}{\partial\theta} = \begin{bmatrix} \frac{\partial z_l}{\partial g}& \frac{\partial z_l}{\partial e}& \frac{\partial z_l}{\partial\mu} \end{bmatrix}. } \$$

Each column tells us how the latent moves when **one physical mechanism changes infinitesimally while everything else is fixed**.

You don't need differentiability through the simulator. Estimate it using symmetric interventions:

\$$ \frac{\partial z}{\partial g} \approx \frac{ z(g+\delta g)-z(g-\delta g) }{ 2\delta g }. \$$

Likewise:

\$$ v_g,\quad v_e,\quad v_\mu. \$$

Now instead of asking whether one subspace exists, we ask whether V-JEPA has constructed a **local causal coordinate chart**.

That gives several falsifiable properties.

---

## 1. Mechanism identifiability

Construct

\$$ J_\theta=[v_g,v_e,v_\mu]. \$$

If

\$$ \operatorname{rank}(J_\theta)=3, \$$

then the three interventions produce locally distinguishable latent effects.

If instead

\$$ \operatorname{rank}(J_\theta)\approx1, \$$

V-JEPA may be collapsing them into something like:

\$$ \text{“amount of motion change.”} \$$

This already distinguishes:

\$$ \boxed{\text{mechanism representation}} \$$

from

\$$ \boxed{\text{generic kinematic representation}}. \$$

---

## 2. Does the causal coordinate system survive context?

This is much deeper.

Estimate

\$$ J_\theta^{(A)} \$$

for a falling sphere.

Then:

\$$ J_\theta^{(B)} \$$

for a bouncing cube.

Then:

\$$ J_\theta^{(C)} \$$

for an object on an inclined plane.

The visual consequences of changing \$g\$ are completely different in these three worlds.

But the underlying causal intervention is identical:

\$$ g\rightarrow g+\delta. \$$

If their causal tangent spaces remain aligned,

\$$ \operatorname{span}J_\theta^{(A)} \approx \operatorname{span}J_\theta^{(B)} \approx \operatorname{span}J_\theta^{(C)}, \$$

you have something much stronger than a probe result.

It means:

> **V-JEPA is using a reusable coordinate system for mechanisms despite radically different manifestations of those mechanisms.**

That begins to look like abstraction.

---

# 3. Compare mechanism geometry against nuisance geometry

Likewise calculate

\$$ J_\eta = \left[ \frac{\partial z}{\partial\text{camera}}, \frac{\partial z}{\partial\text{colour}}, \frac{\partial z}{\partial\text{lighting}} \right]. \$$

Now compare:

\$$ \operatorname{span}(J_\theta) \$$

with

\$$ \operatorname{span}(J_\eta). \$$

Using principal angles.

If deeper layers increasingly produce something like

\$$ \angle( \mathcal U_{\rm mechanism}, \mathcal U_{\rm nuisance} ) \uparrow, \$$

then the network is doing something extremely meaningful:

\$$ \boxed{ \text{separating how the world \emph{works} from how the world \emph{looks}.} } \$$

This would give a much better interpretation of layer progression than simply saying “gravity probe accuracy increases at layer 8.”

---

# 4. There is an even more fundamental quantity

From the Jacobian construct

\$$ \boxed{ G_\theta = J_\theta^T J_\theta. } \$$

This is a \$3\times3\$ matrix.

For

\$$ \theta=(g,e,\mu), \$$

it looks like

\$$ G_\theta= \begin{bmatrix} v_g^Tv_g & v_g^Tv_e & v_g^Tv_\mu\\\ v_e^Tv_g & v_e^Tv_e & v_e^Tv_\mu\\\ v_\mu^Tv_g & v_\mu^Tv_e & v_\mu^Tv_\mu \end{bmatrix}. \$$

This is a **metric induced on physical-parameter space by the V-JEPA representation**.

That's conceptually important.

The diagonal:

\$$ G_{gg}=\\|v_g\\|^2 \$$

says how sensitive the representation is to gravity.

The off-diagonal:

\$$ G_{ge}=v_g^Tv_e \$$

says how similarly V-JEPA represents gravity and restitution changes.

So instead of asking:

> “Where is gravity?”

you ask:

> **“What geometry of physics has the network internally constructed?”**

That's a much more fundamental question.

---

# 5. Then compare this geometry across worlds

Compute:

\$$ G_\theta^{fall}, \quad G_\theta^{bounce}, \quad G_\theta^{incline}, \quad G_\theta^{collision}. \$$

If these matrices are wildly different, V-JEPA's representation of physics is mostly context-dependent.

If instead:

\$$ G_\theta^{fall} \approx G_\theta^{bounce} \approx G_\theta^{incline}, \$$

then different visual worlds induce approximately the **same internal geometry of physical mechanisms**.

That would be a remarkable result.

It would mean the model has partly factored

\$$ \boxed{\text{physics}} \$$

from

\$$ \boxed{\text{particular realization of physics}}. \$$

This is related to causal disentanglement, whose goal is to recover separate generative factors rather than merely statistical features. [arXiv](https://arxiv.org/abs/2310.17325?utm_source=chatgpt.com)

---

# 6. Then test composition

If these are genuinely mechanism coordinates, small interventions should approximately compose.

Take

\$$ \delta\theta= (\delta g,\delta e,0). \$$

The first-order prediction is

\$$ \boxed{ \Delta z_{\rm predicted} = J_\theta\delta\theta = v_g\delta g+v_e\delta e. } \$$

Now actually simulate the world with **both** changed:

\$$ (g+\delta g,e+\delta e). \$$

Encode it and obtain

\$$ \Delta z_{\rm actual}. \$$

Measure

\$$ \boxed{ E_{\rm composition} = \frac{ \\|\Delta z_{\rm actual} - J_\theta\delta\theta\\|^2 }{ \\|\Delta z_{\rm actual}\\|^2 }. } \$$

Small error means:

> The effects of independently changed mechanisms are locally compositional inside the representation.

This is much more consequential than merely finding separate subspaces.

---

# 7. Now scale the interventions

Try:

\$$ \delta, 2\delta, 3\delta. \$$

If

\$$ z(\theta+\delta) \approx z(\theta)+J_\theta\delta \$$

only locally but breaks for larger changes, you've discovered the **curvature** of the physical representation.

The second-order model is:

\$$ z(\theta+\delta) \approx z(\theta) + J_\theta\delta + \frac12\delta^T H_\theta\delta. \$$

Now \$H_\theta\$ tells you how different physical mechanisms interact nonlinearly in representation space.

That leads to a genuinely geometric characterization:

\$$ \boxed{ \text{physical world parameter manifold} \rightarrow \text{V-JEPA latent manifold}. } \$$

This connects naturally to differential-geometric approaches to learned latent spaces, where Jacobians induce local metrics on latent manifolds. [arXiv](https://arxiv.org/pdf/2405.09800?utm_source=chatgpt.com)

---

# 8. This gives a much stronger interpretation of layers

The existing work says roughly:

\$$ \text{physics becomes decodable} \$$

at particular depths. [arXiv](https://arxiv.org/abs/2602.07050?utm_source=chatgpt.com)

Your experiment could instead ask whether a sequence occurs:

### Early layers

\$$ J_\theta\approx J_\eta \$$

Everything is mixed with visual variation.

### Intermediate layers

\$$ \operatorname{rank}J_\theta\uparrow \$$

Physical mechanisms become distinguishable.

### Later-middle layers

\$$ \angle(J_\theta,J_\eta)\uparrow \$$

Physics separates from appearance.

### Deeper layers

\$$ G_\theta(c) \$$

becomes stable across contexts.

And perhaps later still:

\$$ E_{\rm composition}\downarrow. \$$

Now you haven't merely found a **Physics Emergence Zone**.

You have potentially found a sequence:

\$$ \boxed{ \text{pixels} \rightarrow \text{kinematics} \rightarrow \text{mechanism separation} \rightarrow \text{context-independent causal geometry}. } \$$

That would begin to explain **what “deeper representation” means computationally**.

---

# 9. The cause-versus-effect experiment remains crucial

Consider two transformations.

### Same cause, different effects

\$$ g\rightarrow0.5g \$$

in:

- falling object,
- bouncing object,
- pendulum,
- inclined plane.

Observable consequences differ dramatically.

If their latent mechanism directions remain related:

\$$ v_g^{fall} \sim v_g^{bounce} \sim v_g^{pendulum}, \$$

the representation has abstracted away consequences toward the mechanism.

### Different causes, similar effects

Construct gravity and initial-velocity interventions whose rendered trajectory displacement is closely matched.

If:

\$$ v_g\neq v_v, \$$

V-JEPA distinguishes causal origin despite similar effects.

Together those two experiments eliminate both trivial explanations.

---

# 10. This reveals three fundamentally different possible V-JEPAs

And this, I think, is the actual paper.

### Type I — Appearance model

Representations mainly organize visual similarity:

\$$ z\sim\text{pixels/objects}. \$$

### Type II — Kinematic world model

Representations organize:

\$$ x,v,a,\text{trajectory}. \$$

Different causes producing the same motion are represented similarly.

### Type III — Mechanistic world model

Representations organize the **generating mechanisms**:

\$$ g,e,\mu,\ldots \$$

in structures stable across contexts and capable of composition.

Your experiment is essentially trying to determine:

\$$ \boxed{ \text{Which of these objects is V-JEPA?} } \$$

Current work shows physical variables are available in pretrained video representations but explicitly notes that the representation is distributed rather than simply factorized like a classical physics engine. [arXiv](https://arxiv.org/abs/2602.07050?utm_source=chatgpt.com) The broader world-model literature is also increasingly emphasizing that latent states should be judged by what functions they are sufficient for—prediction, counterfactual reasoning, control—not simply by how much information can be decoded from them. [arXiv](https://arxiv.org/abs/2605.01694?utm_source=chatgpt.com)

## The paper I would now formulate

**“From Motion to Mechanism: Does a Video World Model Learn the Causal Geometry of Physical Dynamics?”**

The central mathematical object is no longer a gravity subspace.

It is

\$$ \boxed{ J_\theta^{(l)} = \frac{\partial E_l(F(\theta,s,\eta))} {\partial\theta} } \$$

and its induced geometry

\$$ \boxed{ G_\theta^{(l)} = J_\theta^{(l)T}J_\theta^{(l)}. } \$$

Then the central hypothesis is:

> **As representations deepen, sensitivity to physical mechanisms becomes distinguishable from nuisance variation, increasingly context-invariant, and increasingly compositional.**

If that pattern appears, you're not merely finding another feature inside V-JEPA. You are showing **how a neural world model transforms observations into an internal geometry of the generating world**.

That also gives us a direct bridge back to your abstraction-depth question: **abstraction depth may ultimately be measurable by the degree to which context-specific observational geometry converges toward context-independent mechanism geometry.**