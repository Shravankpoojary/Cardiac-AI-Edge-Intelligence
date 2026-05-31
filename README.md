# Low-cost, portable and power efficient ECG Signal Analysis and Heartbeat sound Acquisition System for Cardiac Assessment

## 📌 Project Overview
This project presents a low-cost, portable, and power-efficient ECG signal analysis and heartbeat monitoring system. Designed for real-time cardiac assessment, the system leverages a **Dual-MCU architecture** and **TinyML** to classify cardiac rhythms directly on the device with clinical-grade accuracy. By capturing both electrical (ECG) and acoustic (PCG) physiological signatures, it provides a comprehensive, cloud-independent point-of-care diagnostic tool.

### The Problem
Traditional ECG monitoring equipment is often expensive, stationary, and focused on a single parameter. While cloud-based IoT solutions exist, they suffer from data privacy vulnerabilities, high power consumption, latency, and absolute dependency on active internet connectivity, making them unviable for remote areas or frontline health screening.

### The Solution
By utilizing **Edge Computing**, this device processes raw ECG and PCG data locally. It eliminates cloud latency, ensures data privacy, runs advanced noise cancellation, and performs real-time machine learning inference for arrhythmia classification entirely offline.

---

## 🚀 Key Features
* **99.77% Model Accuracy:** Achieved through a custom-trained Convolutional Neural Network (CNN) quantized for microcontrollers.
* **Dual-MCU Architecture:** * **STM32 (Primary DSP Core):** Handles high-speed 12-bit ADC sampling, digital signal processing (FIR filtering, noise cancellation), and R-peak detection.
  * **ESP32 (Inference & UI Core):** Manages the TinyML inference engine (TFLite Micro), SPI-based TFT display rendering, user inputs, and power states.
* **Dual-Modality Acquisition:** Simultaneous capture of electrical cardiac paths (ECG) and phonocardiogram acoustic signatures (PCG).
* **Real-time Local Diagnostics:** Direct visual waveform display, audio streaming for auscultation, and immediate color-coded arrhythmia risk categorization on-device.
* **Collaborative Low-Power Modes:** Intelligent inter-MCU sleep/wake management to optimize battery life for field environments.

---

## 🛠️ Technical Stack
* **Hardware:** STM32F446RE (ARM Cortex-M4), ESP32 WROOM-32, AD8232 ECG Frontend, MAX4466 Electret Microphone, TDA1308 Audio Amplifier.
* **Firmware:** Bare-metal register-level configurations (STM32), C++, Arduino IDE / ESP-IDF.
* **AI/ML:** TensorFlow Lite for Microcontrollers (TFLite Micro), Keras, Python, MIT-BIH Arrhythmia Dataset.
* **DSP & Analysis:** Pan-Tompkins QRS Detection Algorithm, Butterworth & FIR Digital Filters, Adaptive Noise Cancellation (ANC).

---

## 📊 Key Operational Parameters

| Parameter | Value | Details / Purpose |
|:---|:---|:---|
| **ECG Sampling Rate** | 360 Hz | Aligned with standard MIT-BIH dataset formatting |
| **PCG Sampling Rate** | 1000 Hz (1 kHz) | Captures structural acoustic S1/S2 heart sounds |
| **STM32 ADC Resolution**| 12-bit (0–4095) | Ensures high-fidelity physiological digitization |
| **ECG Filter Bandpass** | 0.5 Hz – 40 Hz | Removes baseline wander and muscle artifacts |
| **PCG Filter Bandpass** | 20 Hz – 200 Hz | Attenuates high-frequency environmental noise |
| **TinyML Memory Footprint**| < 60 KB RAM | Highly optimized arena size for edge devices |
| **Inter-MCU UART Baud Rate**| 115200 bps | High-speed data streaming to avoid frame drops |
| **Inference Window Length**| 360 Samples | Sliding array used for real-time model classification |

---

## 🔌 Hardware Connections & Wiring Maps

### 1. Sensor Frontends $\rightarrow$ STM32F446RE
| Sensor Module | Sensor Pin | STM32 Pin | Pin Function / Mode |
|:---|:---|:---|:---|
| **AD8232 ECG** | OUTPUT | `PA0` | ADC1_IN0 (Analog ECG Input) |
| **AD8232 ECG** | LO- (Lead Off -) | `PC1` | GPIO_Input (Pull-up) |
| **AD8232 ECG** | LO+ (Lead Off +) | `PC0` | GPIO_Input (Pull-up) |
| **MAX4466 PCG (Chest)**| OUT | `PA1` | ADC1_IN1 (Primary Heart Sound) |
| **MAX4466 PCG (Ambient)**| OUT | `PB0` | ADC1_IN8 (Noise Reference) |
| **TDA1308 Audio Amp** | LINE_IN | `PA4` | DAC_OUT1 (Cleaned Audio Playback) |

### 2. Inter-MCU Link: STM32F446RE $\leftrightarrow$ ESP32
| STM32 Pin | ESP32 Pin | Connection Function |
|:---|:---|:---|
| `PC5` (UART3 TX) | `GPIO16` (RX2) | Transmits continuous filtered ECG/PCG data packets |
| `PB10` (UART3 RX) | `GPIO17` (TX2) | Receives system control flags & UI confirmations |
| `PB1` (Wake Control) | `GPIO5` (Interrupt)| Inter-MCU handshake line for sleep/wake states |
| `GND` | `GND` | **Common Ground** (Mandatory for stable UART communication) |

---

## 📦 Packet Protocol (STM32 $\rightarrow$ ESP32)

To maintain high-speed throughput without dropping real-time samples, data is packed into 6-byte binary frames streamed over UART at 115200 baud:

| Byte 0 (Header) | Byte 1 (Data Type) | Byte 2 (Data MSB) | Byte 3 (Data LSB) | Byte 4 (Checksum) | Byte 5 (Footer) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `0xFC` | `0x01` (ECG) / `0x02` (PCG) | `MSB` | `LSB` | `XOR (Byte 1 to 3)` | `0xFE` |

* **Example Frame (ECG Sample value 2048):** `[0xFC][0x01][0x08][0x00][0x09][0x02]`

---

## 📐 System Architecture & Visuals

### Block Diagram
<p align="center">
  <img width="1488" height="683" alt="Cardiac Analyzer Block Diagram" src="YOUR_BLOCK_DIAGRAM_IMAGE_URL_HERE" />
  <br>
  <em>Figure 1: Comprehensive dual-MCU hardware architecture and signal processing routing pipeline.</em>
</p>

### Hardware Setup Snapshot
<p align="center">
  <img width="4080" height="1836" alt="Cardiac Analyzer Prototype Layout" src="YOUR_HARDWARE_SNAPSHOT_IMAGE_URL_HERE" />
  <br>
  <em>Figure 2: Physical layout showing the STM32 Nucleo, ESP32 processing unit, sensor arrays, and local TFT interface display.</em>
</p>

---

## 🧠 Digital Signal Processing & TinyML Pipeline
