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
* **Dual-MCU Architecture:**
  * **STM32 (Primary DSP Core):** Handles high-speed 12-bit ADC sampling, digital signal processing (FIR filtering, noise cancellation), and R-peak detection.
  * **ESP32 (Inference & UI Core):** Manages the TinyML inference engine (TFLite Micro), SPI-based TFT display rendering, user inputs, and power states.
* **Dual-Modality Acquisition:** Simultaneous capture of electrical cardiac paths (ECG) and phonocardiogram acoustic signatures (PCG).
* **Real-time Local Diagnostics:** Direct visual waveform display, audio streaming for auscultation, and immediate color-coded arrhythmia risk categorization on-device.
* **Collaborative Low-Power Modes:** Intelligent inter-MCU sleep/wake management to optimize battery life for field environments.

---

## 🛠️ Technical Stack
* **Hardware:** STM32F446RE (ARM Cortex-M4), ESP32 WROOM-32, AD8232 ECG Frontend, MAX4466 Electret Microphone, TDA1308 Audio Amplifier, TLV2372 Operational Amplifier.
* **Firmware:** Bare-metal register-level configurations (STM32), C++, Arduino IDE / ESP-IDF.
* **AI/ML:** TensorFlow Lite for Microcontrollers (TFLite Micro), Keras, Python, MIT-BIH Arrhythmia Dataset.
* **DSP & Analysis:** Pan-Tompkins QRS Detection Algorithm, Butterworth & FIR Digital Filters, Adaptive Noise Cancellation (ANC).

---

## 📊 Key Operational Parameters

| Parameter | Value | Details / Purpose |
|:---|:---|:---|
| **ECG Sampling Rate** | 360 Hz (Actual: 371 Hz) | Aligned with standard MIT-BIH dataset formatting |
| **PCG Sampling Rate** | 1000 Hz (1 kHz) | Captures structural acoustic S1/S2 heart sounds |
| **STM32 ADC Resolution**| 12-bit (0–4095) | Ensures high-fidelity physiological digitization |
| **ECG Filter Bandpass** | 0.5 Hz – 40 Hz | Removes baseline wander and muscle artifacts |
| **PCG Filter Bandpass** | 20 Hz – 200 Hz | Attenuates high-frequency environmental noise |
| **TinyML Memory Footprint**| < 60 KB RAM | Highly optimized arena size for edge devices |
| **Inter-MCU UART Baud Rate**| 115200 bps | High-speed data streaming to avoid frame drops |
| **Inference Window Length**| 360 Samples | Sliding array window for real-time model classification |

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
  <img width="750" alt="Screenshot 2026-05-29 145706" src="https://github.com/user-attachments/assets/01c6eeb2-34c7-48be-8a3d-26998c597620" />
  <br>
  <em>Figure 1: Comprehensive dual-MCU hardware architecture and signal processing routing pipeline.</em>
</p>

### Hardware Setup Snapshot
<p align="center">
  <img width="600" alt="Screenshot 2026-05-29 150419" src="https://github.com/user-attachments/assets/99678392-53c6-4065-8da0-6bcb807cd165" />
  <br>
  <em>Figure 2: Physical hardware implementation showing the integrated STM32 Nucleo, ESP32 processing core, sensor arrays, and local TFT interface display.</em>
</p>

### Signal Hardware Interface & Schematic Overview
<p align="center">
  <img width="750" alt="Circuit Schematic & Interface Layout" src="https://github.com/user-attachments/assets/2d73dd68-23a7-462b-9587-d647493cce77" />
  <br>
  <em>Figure 3: Inter-MCU serial routing layout, active amplification schematic, and display interface pinning configuration.</em>
</p>

---

## 🧠 Digital Signal Processing & TinyML Pipeline

Raw Signal ---> Analog Active Filter ---> 12-Bit ADC Sampling ---> Digital FIR Filters ---> Pan-Tompkins Algorithm
|
360-Sample Buffer Window
|
v
Color-Coded Alert Screen <--- On-Device TFT Output <--- Float32 TensorFlow Lite Inference Engine


### 1. Digital Signal Processing (DSP)
* **50 Hz Notch Filter (ECG):** Removes powerline AC hum from the environment. Implemented on the STM32 via an **81-tap FIR configuration** with a stopband window spanning `48 Hz - 52 Hz`.
* **40 Hz Low-Pass Filter (ECG):** Attenuates high-frequency muscle contractions and baseline artifact wander using an **81-tap low-pass filter** cutting off strictly at `40 Hz` to preserve the vital QRS complex.
* **200 Hz Low-Pass Filter (PCG):** Blocks environmental cross-talk while passing the primary **"lub-dub" (S1/S2)** structural heart sounds via an **81-tap architecture** operating at a `1000 Hz` sampling rate.
* **Adaptive Noise Cancellation (ANC):** Leverages a dual-microphone framework. One MAX4466 is encapsulated within the stethoscope chest piece, while a second standalone MAX4466 samples ambient noise to apply dynamic subtraction filtering.
* **Pan-Tompkins Algorithm:** Executes real-time derivative squaring, moving-window integration, and an adaptive thresholding pipeline to isolate individual R-peaks for accurate heart rate calculation.

### 2. TinyML Inference Configuration
* **Dataset & Augmentation:** Trained on the standard **MIT-BIH Arrhythmia Database**. The training profiles were balanced using target noise injection, including Gaussian noise additions ($\sigma = 0.005-0.02$), random scaling ($85\% - 115\%$), and time-shifting to balance classes evenly.
* **Target Classes:** Real-time multi-class classification tracking:
  * **Normal Sinus Rhythm (N)**
  * **Atrial Fibrillation (AFib)**
  * **Atrial Flutter (AFL)**
  * **Premature Ventricular Contraction (PVC)**
* **Edge Quantization:** Full post-training integer-quantization reduces model allocation size, allowing it to execute within a sub-**60KB RAM** arena inside the ESP32's TFLite Micro runtime environment.

---

## 📈 Evaluation & Results

### Model Performance Metrics
<p align="center">
  <img width="700" alt="Screenshot 2026-03-16 160604" src="https://github.com/user-attachments/assets/75f343c0-fdcc-4bc9-8fd8-32ae14c51f46" />
  <br>
  <em>Figure 4: Quantized TinyML model validation results showing Confusion Matrix classification tracking and Epoch training progress.</em>
</p>

<p align="center">
  <img width="700" alt="Screenshot 2026-03-16 165128" src="https://github.com/user-attachments/assets/b7f2906e-4b25-42e1-84ed-f3dec2aa77f6" />
  <br>
  <em>Figure 4: Quantized TinyML model validation results showing Confusion Matrix classification tracking and Epoch training progress.</em>
</p>

### Live Diagnosis Telemetry Walkthrough
<p align="center">
  <img width="750" alt="Screenshot 2026-03-16 231528" src="https://github.com/user-attachments/assets/d4cc39e2-bdb4-4a7b-8277-2be24acae44b" />
  <br>
  <em>Figure 5: Live ECG real-time tracking waveforms along with on-device risk assessment probability matrix indicators.</em>
</p>

### Edge Telemetry Stream & Console Logging
<p align="center">
  <img width="450" alt="Screenshot 2026-03-17 225311" src="https://github.com/user-attachments/assets/548b5051-fadd-460f-84b5-63cc18f4d442" />
  <br>
  <em>Figure 6: High-speed UART frame transmissions showing continuous streaming logs and inference state outcomes.</em>
</p>

<p align="center">
  <img width="450" alt="Screenshot 2026-03-17 224932" src="https://github.com/user-attachments/assets/32c7cdb9-127e-40e5-b0b5-186144adec37" />
  <br>
  <em>Figure 6: High-speed UART frame transmissions showing continuous streaming logs and inference state outcomes.</em>
</p>

<p align="center">
  <img width="450" alt="Screenshot 2026-03-17 225159" src="https://github.com/user-attachments/assets/cc17e351-0f86-4715-a325-4a2c04969496" />
  <br>
  <em>Figure 6: High-speed UART frame transmissions showing continuous streaming logs and inference state outcomes.</em>
</p>

<p align="center">
  <img width="450" alt="Screenshot 2026-03-17 225036" src="https://github.com/user-attachments/assets/2954d73b-c9da-4e05-8a87-9bebcec7e9a1" />
  <br>
  <em>Figure 6: High-speed UART frame transmissions showing continuous streaming logs and inference state outcomes.</em>
</p>

* **High-Fidelity Waves:** Clean real-time ECG trace tracking and dynamic R-peak detection plots rendered directly to the local display module.
* **Validated Classification:** Proven capability to isolate high-probability events on edge devices (e.g., identifying Atrial Fibrillation anomalies with distinct confidence index values during testing loops).
* **Power Savings:** Successfully triggers ultra-low-power idle states via the collaborative inter-MCU sleep timer loop, instantly waking on any button-based hardware interrupt.

---

## ⚙️ Setup & Deployment Instructions

### 1. STM32 Primary (DSP Master Core)
1. Open the firmware directory using **STM32CubeIDE** or **Arduino IDE** (with the `STM32duino` core installed).
2. Configure clock parameters to run the ARM Cortex-M4 core at full processing capacity.
3. Flash the code onto the **Nucleo-F446RE** development board.
4. Open the serial console at `115200 baud` to verify real-time 12-bit ADC digitization and packet streaming output.

### 2. ESP32 Secondary (TinyML & UI Core)
1. Open the ESP32 firmware sketch within the Arduino IDE.
2. Ensure the required libraries are installed:
   * `TensorFlowLite_ESP32` (TFLite Micro Runtime)
   * `Adafruit_ILI9341` & `Adafruit_GFX` (for the SPI-TFT display framework)
3. Confirm that your pre-compiled, quantized model array file (`model_data.h`) is linked inside the same directory.
4. Select **NodeMCU-32S** (or your target ESP32 board variation) and flash the firmware.

---

## 👥 Core Project Team
* **Shravan K Poojary** (USN: 4VP22EC049) - Electronics & Communication Engineering
* **Kaushik NG** (USN: 4VP22EC022) - Electronics & Communication Engineering
* **Charith B** (USN: 4VP22EC013) - Electronics & Communication Engineering
* **Pruthviraj** (USN: 4VP22EC040) - Electronics & Communication Engineering

**Project Mentor:** Prof. Mahabaleshwara Bhat P (Assistant Professor, Dept. of ECE, VCET Puttur)

---

## 🎯 Project Demonstration

<p align="center">
  <b>▶️ CLICK THE IMAGE BELOW TO WATCH THE VIDEO DEMONSTRATION</b><br><br>
  <a href="https://youtu.be/CDvIj5JJK60">  
    <img width="750" alt="Project Demonstration Video Link" src="https://github.com/user-attachments/assets/039580f7-abfa-4294-b891-45bff2db7d14" />
  </a>
  <br>
  <em>YouTube video showing a real-time walkthrough and analytical performance of the Cardiac AI Edge system.</em>
</p>
