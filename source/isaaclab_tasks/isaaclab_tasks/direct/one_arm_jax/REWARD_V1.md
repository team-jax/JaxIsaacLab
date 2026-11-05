# Reward v1: contact geometry grasp, lift, and hold

This experiment freezes the existing 40-observation / 9-action interface, the
0.08 x 0.08 x 0.085 m object, and the 7056-environment scene. It starts from a
new policy. The old finger-1 world-axis target reward was introduced because
earlier evaluations placed finger 1 above the block; it is removed here in
favor of object-relative contact geometry.

References:

- Koenig et al., *The Role of Tactile Sensing in Learning and Deploying Grasp
  Refinement Algorithms*, https://arxiv.org/abs/2109.11234
- Roa and Suárez, *Grasp quality measures: review and performance*,
  https://doi.org/10.1007/s10514-014-9402-3
- Merzic et al., *Leveraging Contact Forces for Learning to Grasp*,
  https://arxiv.org/abs/1809.07004

The papers motivate combining contact locations, directions, and forces. The
specific weights and thresholds below are experiment choices, not constants
claimed by those papers. Isaac Lab supplies the mean contact point and net
normal-force vector for each distal link/object pair; the vector direction is
used as a proxy for a contact normal. This is not a full force-closure metric.

At 60 Hz, per control step:

```text
R = 1.0 reach + 0.5 contact_quality + 0.5 contact_geometry
  + 1.0 stable_grasp_quality + 2.0 lift + 1.0 hold + 10.0 success
  - 0.5 excessive_force - 3.0 new_drop - 0.01 action_rate
```

- `reach = exp(-12*object_to_mean_distal_distance) * (1 - meaningful_contacts/3)`;
  zero after stable grasp.
- A meaningful contact has at least 2 N horizontal normal force, a finite
  contact point in the object's side band (5 mm clear of its top/bottom),
  nonzero radial offset, and radial/force alignment at least 0.35.
- Each contact's force quality rises from 0 to 1 at 8 N, stays 1 to 18 N,
  then falls to 0 at 20 N. The `20 N` normalization is **temporary** and
  remains the fixed software observation convention for this checkpoint.
- `contact_quality` averages force quality times radial/normal alignment over
  three fingers. `contact_geometry` rewards the horizontal directions from the
  object center to contact points being spread around the object. Contacts
  clustered on one side receive little geometry reward.
- `grasp_quality = min(contact_quality, contact_geometry) *
  clamp(1-relative_speed/0.25, 0, 1) * finger_closed`.
  All three contacts and grasp quality above 0.35 must persist for three
  control steps to make a stable grasp.
- Lift is object height gain from 0 to 4 cm, gated by stable grasp. Hold
  requires stable grasp, 4 cm lift, and vertical velocity above -0.05 m/s.
  Thirty continuous control steps (0.5 s) produce success and termination.
- After a stable grasp has lifted 1 cm, quality loss plus a cumulative 5 mm
  descent from the peak and downward velocity below -0.01 m/s counts as one
  drop. Action-rate penalizes the squared change in the nine actions.

The policy sees only three normalized pressure values, not privileged contact
points or normals. Reward v1 uses those simulator quantities only for training
and evaluation. No transport, placement, or release behavior is rewarded.
