# Low-cost, portable and power efficient ECG Signal Analysis and Heartbeat sound Acquisition System for Cardiac Assessmentt 

📌 𝗣𝗿𝗼𝗷𝗲𝗰𝘁 𝗢𝘃𝗲𝗿𝘃𝗶𝗲𝘄:
This project presents a low-cost, portable, and power-efficient ECG signal analysis and heartbeat monitoring system. Designed for real-time cardiac assessment, the system leverages a Dual-MCU architecture and TinyML to classify cardiac rhythms directly on the device with clinical-grade accuracy.

𝗧𝗵𝗲 𝗣𝗿𝗼𝗯𝗹𝗲𝗺:
Traditional ECG monitoring equipment is often expensive and stationary. There is a critical need for affordable, point-of-care diagnostics that can provide real-time alerts without relying on cloud connectivity.

𝗧𝗵𝗲 𝗦𝗼𝗹𝘂𝘁𝗶𝗼𝗻:
By utilizing Edge Computing, this device processes raw ECG data and classifies it into rhythmic categories in real-time. This eliminates latency, ensures data privacy, and allows for deployment in remote areas with limited infrastructure.

🚀 𝗞𝗲𝘆 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀:
99.77% Model Accuracy: Achieved through a custom-trained Convolutional Neural Network (CNN) quantized for microcontrollers.
Dual-MCU Architecture: * STM32: Handles high-speed sampling and real-time digital signal processing (Pan-Tompkins QRS Detection).
ESP32: Manages the AI inference engine (TFLite Micro), OLED display, and communication.
Real-time Diagnostics: Instant classification of heartbeat sounds and ECG waveforms.
Edge-AI (TinyML): Model runs entirely on-device with a memory footprint of less than 60KB RAM.


🛠️ Technical Stack
Hardware: STM32 (ARM Cortex-M), ESP32, ECG Sensor (AD8232), Heartbeat Sound Sensor.
Firmware: C++, Bare-metal register-level programming (STM32), Arduino/ESP-IDF.
AI/ML: TensorFlow Lite for Microcontrollers (TFLite Micro), Keras, Python.
Digital Signal Processing: Pan-Tompkins Algorithm, Butterworth filtering.
