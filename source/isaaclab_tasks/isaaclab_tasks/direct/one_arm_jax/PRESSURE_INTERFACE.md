# Three-finger pressure interface

The policy receives three pressure values in this fixed order:

| Value | Physical location | Simulation source |
| --- | --- | --- |
| `pressure_1` | RA12P on finger 1, second phalanx | `finger_1_link2`–object contact |
| `pressure_2` | RA12P on finger 2, second phalanx | `finger_2_link2`–object contact |
| `pressure_3` | RA12P on finger 3, second phalanx | `finger_3_link2`–object contact |

The policy input for each finger is dimensionless and uses the same conversion in simulation and on STM32:

```text
pressure_i = clamp(force_i_newtons / 20.0, 0.0, 1.0)
```

Thus zero means no measured load and one means at least 20 N. The 20 N value is a **software scale for the policy**, not the RA12P hardware limit. The supplier lists a sensing range of 5 g to 4 kgf (about 0.05 to 39 N). Current policy thresholds are 0.10 for contact (2 N), 0.40 for desired pressure (8 N), and 0.90 for excess pressure (18 N). These reward thresholds can be reviewed using grasp traces; the three input channels and their normalization stay fixed for a policy checkpoint.

In simulation, Isaac Lab reads contact forces on each second phalanx. The current approximation uses the horizontal magnitude of each link–object contact force so that pushing down on the object's top is not counted as side grasp pressure. It measures the **whole link**, not the RA12P's 4.8 mm sensing area; simulated values are not hardware-calibrated force readings.

When the hardware arrives, measure several known loads on **each installed sensor**, with the actual STM32 ADC and voltage-divider circuit. Fit an ADC-to-newton lookup table or curve for each finger, including its unloaded offset. Convert the resulting newtons with the formula above and publish `pressure_1`, `pressure_2`, `pressure_3` in that order. Do not map raw ADC counts linearly to 0–1 without calibration: an FSR's electrical response and the circuit determine that relationship.

The policy observation still has exactly three pressure channels. No ADC counts or extra pressure history are added to the observation.
