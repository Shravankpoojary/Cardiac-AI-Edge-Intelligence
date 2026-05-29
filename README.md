# Low-cost, portable and power efficient ECG Signal Analysis and Heartbeat sound Acquisition System for Cardiac Assessment

## 📌 Project Overview
This project presents a low-cost, portable, and power-efficient ECG signal analysis and heartbeat monitoring system[cite: 80]. Designed for real-time cardiac assessment, the system leverages a **Dual-MCU architecture** and **TinyML** to classify cardiac rhythms directly on the device with clinical-grade accuracy[cite: 11, 378]. By capturing both electrical (ECG) and acoustic (PCG) physiological signatures [cite: 373], it provides a comprehensive, cloud-independent point-of-care diagnostic tool[cite: 11, 12].

### The Problem
Traditional ECG monitoring equipment is often expensive, stationary, and focused on a single parameter[cite: 29, 31]. While cloud-based IoT solutions exist, they suffer from data privacy vulnerabilities, high power consumption, latency, and absolute dependency on active internet connectivity [cite: 30], making them unviable for remote areas or frontline health screening[cite: 306].

### The Solution
By utilizing **Edge Computing**, this device processes raw ECG and PCG data locally[cite: 11, 373]. It eliminates cloud latency, ensures data privacy [cite: 11], runs advanced noise cancellation [cite: 377], and performs real-time machine learning inference for arrhythmia classification entirely offline[cite: 11, 130].

---

## 🚀 Key Features
* **99.77% Model Accuracy:** Achieved through a custom-trained Convolutional Neural Network (CNN) quantized for microcontrollers.
* **Dual-MCU Architecture:** * **STM32 (Primary DSP Core):** Handles high-speed 12-bit ADC sampling, digital signal processing (FIR filtering, noise cancellation), and R-peak detection[cite: 100, 107, 108].
  * **ESP32 (Inference & UI Core):** Manages the TinyML inference engine (TFLite Micro), SPI-based TFT display rendering, user inputs, and power states[cite: 116, 118, 125].
* **Dual-Modality Acquisition:** Simultaneous capture of electrical cardiac paths (ECG) and phonocardiogram acoustic signatures (PCG)[cite: 373].
* **Real-time Local Diagnostics:** Direct visual waveform display, audio streaming for auscultation [cite: 41], and immediate color-coded arrhythmia risk categorization on-device[cite: 129].
* **Collaborative Low-Power Modes:** Intelligent inter-MCU sleep/wake management to optimize battery life for field environments[cite: 120, 296].

---

## 🛠️ Technical Stack
* **Hardware:** STM32F446RE (ARM Cortex-M4) [cite: 227], ESP32 WROOM-32 [cite: 232], AD8232 ECG Frontend [cite: 382], MAX4466 Electret Microphone [cite: 385], TDA1308 Audio Amplifier[cite: 233].
* **Firmware:** Bare-metal register-level configurations (STM32), C++, Arduino IDE / ESP-IDF[cite: 86].
* **AI/ML:** TensorFlow Lite for Microcontrollers (TFLite Micro) [cite: 126], Keras, Python, MIT-BIH Arrhythmia Dataset[cite: 299].
* **DSP & Analysis:** Pan-Tompkins QRS Detection Algorithm [cite: 108], Butterworth & FIR Digital Filters [cite: 86, 384], Adaptive Noise Cancellation (ANC)[cite: 387].

---

## 📐 System Architecture & Methodology

The device utilizes a parallel-processing approach split across a dedicated signal conditioning pipeline and an application management core[cite: 381].

### Block Diagram
