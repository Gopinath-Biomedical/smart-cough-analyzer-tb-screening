# Hardware

## Components (from project design)

| Component | Notes |
|---|---|
| ESP32 dev board | Runs TFLite Micro inference |
| Electret microphone module | Analog or I2S input (an I2S MEMS mic such as INMP441 is an option for better quality) |
| OLED display | I2C, shows result |
| Li-polymer battery 3.7 V | With charging module |

Target total cost: under ₹2000. Add your actual part prices below.

| Item | Qty | Cost (₹) |
|---|---|---|
| ESP32 | 1 | |
| Mic module | 1 | |
| OLED | 1 | |
| Battery + charger | 1 | |
| Enclosure / misc | 1 | |
| **Total** | | |

## Block diagram

See [images/hardware_design.jpg](../images/hardware_design.jpg).

Mic -> ESP32 (noise filtering, MFCC, CNN) -> OLED

## To add

- Circuit / wiring diagram (Fritzing or KiCad)
- Photos of the prototype
- ESP32 firmware (`firmware/` folder) once ready
