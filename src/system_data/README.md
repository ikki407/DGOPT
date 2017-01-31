# Distribution System Data

## File Format

`.csv`

## Columns

| Name | Description | 
|:-----------|:------------:|
| Node | Bus number |
| P | Active load [W] |
| Q | Reactive load [W] |
| R | Resistance [Ω] |
| X | Reactance [Ω] |
| candidate\_PV | Binary parameter if the bus is the candidate one of PV installation. |
| candidate\_wd | Binary parameter if the bus is the candidate one of wind turbine installation. |
| candidate\_CB | Binary parameter if the bus is the candidate one of capasitor bank installation. |

## Reference

[1]M Chis, MMA Salama, and S Jayaram. Capacitor placement in distribution systems using heuristic search strategies. IEE Proceedings-Generation, Transmission and Distribution, Vol. 144, No. 3, pp. 225–230, 1997.

[2]D Das, DP Kothari, and A Kalam. Simple and efficient method for load flow solution of radial distribution networks. International Journal of Electrical Power & Energy Systems, Vol. 17, No. 5, pp. 335–346, 1995.

