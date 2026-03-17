/*
 * STM32 ECG/PCG + Pan-Tompkins QRS Detection - COMBINED SYSTEM
 * 
 * Features:
 * - 360Hz ECG sampling with Pan-Tompkins QRS detection
 * - 1000Hz PCG sampling with ANC
 * - Binary UART protocol: ECG samples + Heart Rate transmission
 * - BPM jump protection and multi-stage validation
 * - 5-minute sleep/wake synchronization with ESP32
 * - Wake on D13 (PB1) LOW pulse from ESP32 GPIO5
 * 
 * UART Protocol:
 * ECG Sample: [0xAA][MSB][LSB][0x55]
 * Heart Rate: [0xBB][BPM_HIGH][BPM_LOW][0x66]
 */

#include "CMSIS_DSP.h"
#include <STM32TimerInterrupt.h>
#include <arm_math.h>

// ============ UART CONFIGURATION ============
HardwareSerial Serial3(PC5, PB10);

// ============ DUAL TIMERS ============
STM32Timer ECG_Timer(TIM1);
STM32Timer PCG_Timer(TIM2);

// ============ HARDWARE DEFINITIONS ============
const int ecgAnalogPin = PA0;
const int heartMicPin = PA1;
const int noiseMicPin = PB0;
const int pcgDacPin = PA4;

// ============ SLEEP/WAKE CONFIGURATION ============
#define WAKEUP_PIN 13  // D13 - Connected to ESP32 GPIO5
#define INACTIVITY_TIMEOUT 150000  // 2.5 minutes

volatile uint32_t lastActivityTime = 0;
volatile bool systemActive = true;
volatile bool wakeupTriggered = false;

// ============ ECG SAMPLING CONFIGURATION ============
const double ECG_SAMPLING_FREQ = 360.0;
const double ECG_SAMPLE_INTERVAL_US = 1000000.0 / ECG_SAMPLING_FREQ;

volatile uint16_t ecg_rawSample = 0;
volatile uint16_t ecg_filteredValue = 0;

// ============ PAN-TOMPKINS CONFIGURATION ============
#define WINDOWSIZE 20
#define NOSAMPLE -32000
#define FS 360
#define BUFFSIZE 600
#define DELAY 22

typedef int dataType;
typedef enum {FALSE, TRUE} BOOL;

// ============ VALIDATION PARAMETERS ============
#define MIN_RR_MS 300           // 200 BPM max
#define MAX_RR_MS 2000          // 30 BPM min  
#define REFRACTORY_MS 200       // Absolute refractory period
#define MAX_BPM_JUMP 25         // Max allowed BPM change
#define MAX_RR_JUMP_MS 300      // Tighter RR jump limit

// ============ AD8232-SPECIFIC PARAMETERS ============
#define INITIAL_CALIBRATION_SAMPLES 200
#define MIN_QRS_AMPLITUDE 40
#define SIGNAL_TO_NOISE_RATIO 2.5
#define ADAPTIVE_LEARNING_RATE 0.125

// ============ BPM STABILIZATION ============
#define BPM_HISTORY_SIZE 8
#define MAX_BPM_VARIATION 15

// ============ PAN-TOMPKINS GLOBAL VARIABLES ============
volatile dataType currentECGSample = 0;
volatile bool newSampleReady = false;
int qrsDetectedFlag = 0;

unsigned long lastBeatTime = 0;
unsigned long lastValidBeatTime = 0;
int beatCount = 0;
int validBeatCount = 0;
bool firstBeat = true;
int currentBPM = 0;
int stableBPM = 70;

// ============ BPM TRANSMISSION ============
volatile int transmitBPM = 0;
volatile bool bpmUpdateReady = false;
volatile bool rpeakMarkerReady = false;

// ============ BPM STABILIZATION STRUCTURE ============
struct BPMStabilizer {
    int history[BPM_HISTORY_SIZE];
    int historyIndex;
    int historyCount;
    int lastStableBPM;
    bool signalStable;
    float variation;
};

BPMStabilizer bpmStabilizer = {{0}, 0, 0, 70, false, 0.0};

// ============ AMPLITUDE TRACKING ============
struct AmplitudeTracker {
    dataType recentQRSAmplitudes[16];
    int amplitudeIndex;
    int amplitudeCount;
    dataType currentQRSAmplitude;
    dataType avgQRSAmplitude;
    dataType noiseFloor;
    bool calibrationComplete;
};

AmplitudeTracker amplitudeTracker = {{0}, 0, 0, 0, 0, 0, false};

// ============ CIRCULAR BUFFER FOR RR INTERVALS ============
#define RR_BUFFER_SIZE 32
struct CircularBuffer {
    uint32_t data[RR_BUFFER_SIZE];
    int head;
    int count;
};

CircularBuffer rrBuffer = {{0}, 0, 0};

// ============ SIGNAL QUALITY ============
struct SignalQuality {
    int totalBeats;
    int validBeats;
    int qualityScore;
    int consecutiveRejects;
    bool highNoiseMode;
};

SignalQuality signalQuality = {0, 0, 100, 0, false};

// ============ ECG FILTERING ============
arm_fir_instance_f32 ecg_fir_notch_50hz;
arm_fir_instance_f32 ecg_fir_lowpass;
arm_fir_instance_f32 ecg_fir_highpass;

const float32_t ecg_notch_coeffs[81] = {
    0.0012, 0.0011, 0.0002, -0.0010, -0.0017, -0.0013, 0.0004, 0.0023,
    0.0028, 0.0012, -0.0020, -0.0044, -0.0039, 0.0000, 0.0049, 0.0070,
    0.0039, -0.0029, -0.0088, -0.0088, -0.0019, 0.0076, 0.0127, 0.0087,
    -0.0025, -0.0131, -0.0150, -0.0057, 0.0087, 0.0179, 0.0144, 0.0000,
    -0.0153, -0.0202, -0.0105, 0.0073, 0.0203, 0.0189, 0.0038, -0.0142,
    0.9758, -0.0142, 0.0038, 0.0189, 0.0203, 0.0073, -0.0105, -0.0202,
    -0.0153, 0.0000, 0.0144, 0.0179, 0.0087, -0.0057, -0.0150, -0.0131,
    -0.0025, 0.0087, 0.0127, 0.0076, -0.0019, -0.0088, -0.0088, -0.0029,
    0.0039, 0.0070, 0.0049, 0.0000, -0.0039, -0.0044, -0.0020, 0.0012,
    0.0028, 0.0023, 0.0004, -0.0013, -0.0017, -0.0010, 0.0002, 0.0011,
    0.0012
};

const float32_t ecg_lowpass_coeffs[81] = {
    0.0002, 0.0006, 0.0007, 0.0005, -0.0000, -0.0007, -0.0012, -0.0012,
    -0.0006, 0.0007, 0.0020, 0.0026, 0.0020, -0.0000, -0.0026, -0.0046,
    -0.0046, -0.0020, 0.0023, 0.0066, 0.0084, 0.0062, -0.0000, -0.0078,
    -0.0133, -0.0131, -0.0058, 0.0065, 0.0186, 0.0239, 0.0177, -0.0000,
    -0.0233, -0.0417, -0.0436, -0.0210, 0.0266, 0.0906, 0.1556, 0.2040,
    0.2218, 0.2040, 0.1556, 0.0906, 0.0266, -0.0210, -0.0436, -0.0417,
    -0.0233, -0.0000, 0.0177, 0.0239, 0.0186, 0.0065, -0.0058, -0.0131,
    -0.0133, -0.0078, -0.0000, 0.0062, 0.0084, 0.0066, 0.0023, -0.0020,
    -0.0046, -0.0046, -0.0026, -0.0000, 0.0020, 0.0026, 0.0020, 0.0007,
    -0.0006, -0.0012, -0.0012, -0.0007, -0.0000, 0.0005, 0.0007, 0.0006,
    0.0002
};

const float32_t ecg_highpass_coeffs[81] = {
    -0.00021778f, -0.00022187f, -0.00023367f, -0.00025313f, -0.00028016f,
    -0.00031461f, -0.00035629f, -0.00040495f, -0.00046032f, -0.00052206f,
    -0.00058981f, -0.00066316f, -0.00074167f, -0.00082486f, -0.00091223f,
    -0.00100323f, -0.00109732f, -0.00119392f, -0.00129242f, -0.00139223f,
    -0.00149272f, -0.00159328f, -0.00169328f, -0.00179209f, -0.00188912f,
    -0.00198374f, -0.00207538f, -0.00216346f, -0.00224744f, -0.00232678f,
    -0.00240100f, -0.00246963f, -0.00253224f, -0.00258844f, -0.00263788f,
    -0.00268025f, -0.00271528f, -0.00274276f, -0.00276251f, -0.00277441f,
     0.99743929f, // Center tap (Index 40)
    -0.00277441f, -0.00276251f, -0.00274276f, -0.00271528f, -0.00268025f,
    -0.00263788f, -0.00258844f, -0.00253224f, -0.00246963f, -0.00240100f,
    -0.00232678f, -0.00224744f, -0.00216346f, -0.00207538f, -0.00198374f,
    -0.00188912f, -0.00179209f, -0.00169328f, -0.00159328f, -0.00149272f,
    -0.00139223f, -0.00129242f, -0.00119392f, -0.00109732f, -0.00100323f,
    -0.00091223f, -0.00082486f, -0.00074167f, -0.00066316f, -0.00058981f,
    -0.00052206f, -0.00046032f, -0.00040495f, -0.00035629f, -0.00031461f,
    -0.00028016f, -0.00025313f, -0.00023367f, -0.00022187f, -0.00021778f
};
#define ECG_NOTCH_TAPS 81
#define ECG_LOWPASS_TAPS 81
#define BLOCK_SIZE 1

float32_t ecg_notch_state[ECG_NOTCH_TAPS + BLOCK_SIZE - 1] = {0};
float32_t ecg_lowpass_state[ECG_LOWPASS_TAPS + BLOCK_SIZE - 1] = {0};
float32_t ecg_highpass_state[81 + BLOCK_SIZE - 1] = {0};
float32_t ecg_inputBuffer[1], ecg_notchOutput[1], ecg_lowpassOutput[1], ecg_highpassOutput[1];


// ============ PCG CONFIGURATION ============
const double PCG_SAMPLING_FREQ = 1000.0;
const double PCG_SAMPLE_INTERVAL_US = 1000000.0 / PCG_SAMPLING_FREQ;

volatile uint16_t pcg_rawHeartSample = 0;
volatile uint16_t pcg_rawNoiseSample = 0;
volatile uint16_t pcg_filteredValue = 0;

arm_fir_instance_f32 pcg_fir_notch_50hz;
arm_fir_instance_f32 pcg_fir_lowpass;

const float32_t pcg_lowpass_coeffs[81] = {
  -0.00000000f,  -0.00053825f,  -0.00068326f,  -0.00024676f,
  0.00053342f,  0.00104727f,  0.00071698f,  -0.00044103f,
  -0.00158984f,  -0.00158200f,  0.00000000f,  0.00214488f,
  0.00291855f,  0.00109296f,  -0.00238593f,  -0.00463969f,
  -0.00310526f,  0.00185236f,  0.00644816f,  0.00618670f,
  -0.00000000f,  -0.00781743f,  -0.01030447f,  -0.00375029f,
  0.00798631f,  0.01521288f,  0.01001950f,  -0.00591149f,
  -0.02046838f,  -0.01965914f,  0.00000000f,  0.02549001f,
  0.03455815f,  0.01311502f,  -0.02965496f,  -0.06150626f,
  -0.04577603f,  0.03240969f,  0.15068801f,  0.25746032f,
  0.30035859f,  0.25746032f,  0.15068801f,  0.03240969f,
  -0.04577603f,  -0.06150626f,  -0.02965496f,  0.01311502f,
  0.03455815f,  0.02549001f,  0.00000000f,  -0.01965914f,
  -0.02046838f,  -0.00591149f,  0.01001950f,  0.01521288f,
  0.00798631f,  -0.00375029f,  -0.01030447f,  -0.00781743f,
  -0.00000000f,  0.00618670f,  0.00644816f,  0.00185236f,
  -0.00310526f,  -0.00463969f,  -0.00238593f,  0.00109296f,
  0.00291855f,  0.00214488f,  0.00000000f,  -0.00158200f,
  -0.00158984f,  -0.00044103f,  0.00071698f,  0.00104727f,
  0.00053342f,  -0.00024676f,  -0.00068326f,  -0.00053825f,
  -0.00000000f
};

const float32_t pcg_notch_coeffs[81] = {
  -0.00061377f,  -0.00059533f,  -0.00053392f,  -0.00042067f,
  -0.00024502f,  0.00000000f,  0.00031221f,  0.00067560f,
  0.00105797f,  0.00141177f,  0.00167848f,  0.00179632f,
  0.00171028f,  0.00138302f,  0.00080469f,  -0.00000000f,
  -0.00096930f,  -0.00200732f,  -0.00299263f,  -0.00379197f,
  -0.00427731f,  -0.00434432f,  -0.00392943f,  -0.00302295f,
  -0.00167606f,  0.00000000f,  0.00184283f,  0.00365538f,
  0.00522828f,  0.00636519f,  0.00690822f,  0.00675966f,
  0.00589725f,  0.00438059f,  0.00234742f,  0.00000000f,
  -0.00241696f,  -0.00464429f,  -0.00643879f,  -0.00760217f,
  0.99261404f,  -0.00760217f,  -0.00643879f,  -0.00464429f,
  -0.00241696f,  0.00000000f,  0.00234742f,  0.00438059f,
  0.00589725f,  0.00675966f,  0.00690822f,  0.00636519f,
  0.00522828f,  0.00365538f,  0.00184283f,  0.00000000f,
  -0.00167606f,  -0.00302295f,  -0.00392943f,  -0.00434432f,
  -0.00427731f,  -0.00379197f,  -0.00299263f,  -0.00200732f,
  -0.00096930f,  -0.00000000f,  0.00080469f,  0.00138302f,
  0.00171028f,  0.00179632f,  0.00167848f,  0.00141177f,
  0.00105797f,  0.00067560f,  0.00031221f,  0.00000000f,
  -0.00024502f,  -0.00042067f,  -0.00053392f,  -0.00059533f,
  -0.00061377f
};

#define PCG_NOTCH_TAPS 81
#define PCG_LOWPASS_TAPS 81

float32_t pcg_notch_state[PCG_NOTCH_TAPS + BLOCK_SIZE - 1] = {0};
float32_t pcg_lowpass_state[PCG_LOWPASS_TAPS + BLOCK_SIZE - 1] = {0};
float32_t pcg_heartInputBuffer[1], pcg_noiseInputBuffer[1];
float32_t pcg_notchOutput[1], pcg_lowpassOutput[1];

#define VOLUME_BOOST_FACTOR 3.9f

// ============ PCG ANC SYSTEM ============
#define ANC_FILTER_LENGTH 32
#define MU 0.01f

float32_t anc_coeffs[ANC_FILTER_LENGTH] = {0};
float32_t noise_buffer[ANC_FILTER_LENGTH] = {0};
bool ancEnabled = true;

float32_t applyANC(float32_t heartSignal, float32_t noiseReference) {
  if (!ancEnabled) return heartSignal;
  
  for (int i = ANC_FILTER_LENGTH - 1; i > 0; i--) {
    noise_buffer[i] = noise_buffer[i - 1];
  }
  noise_buffer[0] = noiseReference;
  
  float32_t estimatedNoise = 0.0f;
  for (int i = 0; i < ANC_FILTER_LENGTH; i++) {
    estimatedNoise += anc_coeffs[i] * noise_buffer[i];
  }
  
  float32_t cleanSignal = heartSignal - estimatedNoise;
  float32_t ancError = cleanSignal;
  
  for (int i = 0; i < ANC_FILTER_LENGTH; i++) {
    anc_coeffs[i] = anc_coeffs[i] + MU * ancError * noise_buffer[i];
    if (anc_coeffs[i] > 1.0f) anc_coeffs[i] = 1.0f;
    if (anc_coeffs[i] < -1.0f) anc_coeffs[i] = -1.0f;
  }
  
  return cleanSignal;
}

bool validateNoiseReference(float32_t heart, float32_t noise) {
  static float32_t heartAvg = 0.0f, noiseAvg = 0.0f;
  static uint32_t sampleCount = 0;
  
  heartAvg = 0.999f * heartAvg + 0.001f * fabs(heart);
  noiseAvg = 0.999f * noiseAvg + 0.001f * fabs(noise);
  sampleCount++;
  
  if (sampleCount > 1000 && noiseAvg < 0.01f) return false;
  return true;
}

// ============ CIRCULAR BUFFER FOR UART ============
#define UART_BUFFER_SIZE 512
volatile uint16_t uart_ecg_buffer[UART_BUFFER_SIZE];
volatile uint16_t uart_write_index = 0;
volatile uint16_t uart_read_index = 0;
volatile uint32_t droppedSamples = 0;

inline void uart_push(uint16_t value) {
  uint16_t next = (uart_write_index + 1) % UART_BUFFER_SIZE;
  if (next != uart_read_index) {
    uart_ecg_buffer[uart_write_index] = value;
    uart_write_index = next;
  } else {
    droppedSamples++;
  }
}

inline bool uart_pop(uint16_t* value) {
  if (uart_read_index == uart_write_index) return false;
  *value = uart_ecg_buffer[uart_read_index];
  uart_read_index = (uart_read_index + 1) % UART_BUFFER_SIZE;
  return true;
}

inline uint16_t uart_buffer_level() {
  return (uart_write_index - uart_read_index + UART_BUFFER_SIZE) % UART_BUFFER_SIZE;
}

// ============ BPM STABILIZATION FUNCTIONS ============
void updateBPMStabilizer(int newBPM) {
    bpmStabilizer.history[bpmStabilizer.historyIndex] = newBPM;
    bpmStabilizer.historyIndex = (bpmStabilizer.historyIndex + 1) % BPM_HISTORY_SIZE;
    
    if (bpmStabilizer.historyCount < BPM_HISTORY_SIZE) {
        bpmStabilizer.historyCount++;
    }
    
    int sum = 0;
    for (int i = 0; i < bpmStabilizer.historyCount; i++) {
        sum += bpmStabilizer.history[i];
    }
    int mean = sum / bpmStabilizer.historyCount;
    
    float variance = 0;
    for (int i = 0; i < bpmStabilizer.historyCount; i++) {
        variance += pow(bpmStabilizer.history[i] - mean, 2);
    }
    bpmStabilizer.variation = sqrt(variance / bpmStabilizer.historyCount);
    
    bpmStabilizer.signalStable = (bpmStabilizer.variation < MAX_BPM_VARIATION);
    bpmStabilizer.lastStableBPM = mean;
}

bool validateBPMJump(int newBPM) {
    if (bpmStabilizer.historyCount < 3) return true;
    
    int lastBPM = bpmStabilizer.lastStableBPM;
    int bpmDiff = abs(newBPM - lastBPM);
    
    int maxAllowedJump = bpmStabilizer.signalStable ? MAX_BPM_JUMP : (MAX_BPM_JUMP * 2);
    
    if (bpmDiff > maxAllowedJump) {
        return false;
    }
    
    return true;
}

// ============ AMPLITUDE VALIDATION ============
void updateAmplitudeTracker(dataType newAmplitude) {
    amplitudeTracker.recentQRSAmplitudes[amplitudeTracker.amplitudeIndex] = newAmplitude;
    amplitudeTracker.amplitudeIndex = (amplitudeTracker.amplitudeIndex + 1) % 16;
    
    if (amplitudeTracker.amplitudeCount < 16) {
        amplitudeTracker.amplitudeCount++;
    }
    
    dataType sum = 0;
    for (int i = 0; i < amplitudeTracker.amplitudeCount; i++) {
        sum += amplitudeTracker.recentQRSAmplitudes[i];
    }
    amplitudeTracker.avgQRSAmplitude = sum / amplitudeTracker.amplitudeCount;
    
    amplitudeTracker.noiseFloor = amplitudeTracker.avgQRSAmplitude * 0.3;
}

bool validateAmplitude(dataType amplitude) {
    if (!amplitudeTracker.calibrationComplete) return true;
    
    if (amplitude < MIN_QRS_AMPLITUDE) {
        return false;
    }
    
    if (amplitude < amplitudeTracker.noiseFloor * SIGNAL_TO_NOISE_RATIO) {
        return false;
    }
    
    if (amplitudeTracker.amplitudeCount > 3) {
        float ratio = (float)amplitude / amplitudeTracker.avgQRSAmplitude;
        if (ratio > 2.0 || ratio < 0.5) {
            return false;
        }
    }
    
    return true;
}

// ============ CIRCULAR BUFFER FUNCTIONS ============
void pushToCircular(CircularBuffer* buf, uint32_t value) {
    buf->data[buf->head] = value;
    buf->head = (buf->head + 1) % RR_BUFFER_SIZE;
    if (buf->count < RR_BUFFER_SIZE) {
        buf->count++;
    }
}

uint32_t getFromCircular(CircularBuffer* buf, int index) {
    if (index >= buf->count) return 0;
    int pos = (buf->head - buf->count + index + RR_BUFFER_SIZE) % RR_BUFFER_SIZE;
    return buf->data[pos];
}

// ============ VALIDATION CHECKS ============
bool passesRefractoryCheck(unsigned long currentTime) {
    if (lastValidBeatTime == 0) return true;
    unsigned long interval = currentTime - lastValidBeatTime;
    return interval >= REFRACTORY_MS;
}

bool passesRangeCheck(unsigned long rrInterval) {
    return (rrInterval >= MIN_RR_MS && rrInterval <= MAX_RR_MS);
}

bool passesJumpCheck(uint32_t newRR) {
    if (rrBuffer.count == 0) return true;
    
    uint32_t lastRR = getFromCircular(&rrBuffer, rrBuffer.count - 1);
    long diff = abs((long)newRR - (long)lastRR);
    
    return diff <= MAX_RR_JUMP_MS;
}

// ============ SIGNAL QUALITY MONITORING ============
void updateSignalQuality(bool beatValid) {
    signalQuality.totalBeats++;
    
    if (beatValid) {
        signalQuality.validBeats++;
        signalQuality.consecutiveRejects = 0;
    } else {
        signalQuality.consecutiveRejects++;
    }
    
    signalQuality.qualityScore = (100 * signalQuality.validBeats) / max(1, signalQuality.totalBeats);
    signalQuality.highNoiseMode = (signalQuality.consecutiveRejects > 5);
    
    if (signalQuality.totalBeats > 100) {
        signalQuality.totalBeats /= 2;
        signalQuality.validBeats /= 2;
    }
}

// ============ RR INTERVAL PROCESSING ============
void processRR(uint32_t rr, unsigned long currentTime, dataType qrsAmplitude) {
    // Validation stages
    if (!passesRefractoryCheck(currentTime)) {
        updateSignalQuality(false);
        return;
    }
    
    if (!passesRangeCheck(rr)) {
        updateSignalQuality(false);
        return;
    }
    
    int newBPM = 60000 / rr;
    if (!validateBPMJump(newBPM)) {
        updateSignalQuality(false);
        return;
    }
    
    if (!passesJumpCheck(rr)) {
        updateSignalQuality(false);
        return;
    }
    
    if (!validateAmplitude(qrsAmplitude)) {
        updateSignalQuality(false);
        return;
    }
    
    // Beat accepted
    validBeatCount++;
    lastValidBeatTime = currentTime;
    
    updateAmplitudeTracker(qrsAmplitude);
    pushToCircular(&rrBuffer, rr);
    
    currentBPM = newBPM;
    updateBPMStabilizer(currentBPM);
    
    // Signal BPM ready for transmission
    transmitBPM = bpmStabilizer.lastStableBPM;
    bpmUpdateReady = true;
    
    updateSignalQuality(true);
}

// ============ PAN-TOMPKINS CORE ============
void initPT() {
    amplitudeTracker.amplitudeIndex = 0;
    amplitudeTracker.amplitudeCount = 0;
    amplitudeTracker.avgQRSAmplitude = 0;
    amplitudeTracker.noiseFloor = 20;
    amplitudeTracker.calibrationComplete = false;
}

dataType inputPT() {
    return currentECGSample;
}

void outputPT(int out) {
    if (out == 1) {
        qrsDetectedFlag = 1;
    }
}

void panTompkins() {
    static dataType signal[BUFFSIZE], dcblock[BUFFSIZE], lowpass[BUFFSIZE], 
                    highpass[BUFFSIZE], derivative[BUFFSIZE], squared[BUFFSIZE], 
                    integral[BUFFSIZE], outputSignal[BUFFSIZE];
    static int rr1[8], rr2[8], rravg1, rravg2, rrlow = 0, rrhigh = 0, rrmiss = 0;
    static long unsigned int sample = 0, lastQRS = 0, lastSlope = 0, currentSlope = 0;
    static dataType peak_i = 0, peak_f = 0, threshold_i1 = 0, threshold_i2 = 0, 
                    threshold_f1 = 0, threshold_f2 = 0, spk_i = 0, spk_f = 0, 
                    npk_i = 0, npk_f = 0;
    static BOOL regular = TRUE, prevRegular;
    
    static dataType signalPeak = 0;
    static int calibrationCounter = 0;
    
    int current;
    BOOL qrs;
    long unsigned int i, j;

    if (sample == 0) {
        for (i = 0; i < 8; i++) {
            rr1[i] = 0;
            rr2[i] = 0;
        }
    }

    if (sample >= BUFFSIZE) {
        for (i = 0; i < BUFFSIZE - 1; i++) {
            signal[i] = signal[i+1];
            dcblock[i] = dcblock[i+1];
            lowpass[i] = lowpass[i+1];
            highpass[i] = highpass[i+1];
            derivative[i] = derivative[i+1];
            squared[i] = squared[i+1];
            integral[i] = integral[i+1];
            outputSignal[i] = outputSignal[i+1];
        }
        current = BUFFSIZE - 1;
    } else {
        current = sample;
    }
    
    signal[current] = inputPT();
    if (signal[current] == NOSAMPLE) return;
    sample++;

    // ========== AUTOMATIC CALIBRATION ==========
    if (!amplitudeTracker.calibrationComplete && sample < INITIAL_CALIBRATION_SAMPLES) {
        calibrationCounter++;
        
        if (abs(signal[current]) > signalPeak) {
            signalPeak = abs(signal[current]);
        }
        
        if (calibrationCounter == INITIAL_CALIBRATION_SAMPLES) {
            threshold_i1 = signalPeak * 0.3;
            threshold_f1 = signalPeak * 0.3;
            npk_i = signalPeak * 0.1;
            npk_f = signalPeak * 0.1;
            spk_i = signalPeak * 0.5;
            spk_f = signalPeak * 0.5;
            
            amplitudeTracker.noiseFloor = signalPeak * 0.1;
            amplitudeTracker.calibrationComplete = true;
        }
    }

    // DC Block filter
    if (current >= 1)
        dcblock[current] = signal[current] - signal[current-1] + 0.995*dcblock[current-1];
    else
        dcblock[current] = 0;

    // Low Pass filter
    lowpass[current] = dcblock[current];
    if (current >= 1) lowpass[current] += 2*lowpass[current-1];
    if (current >= 2) lowpass[current] -= lowpass[current-2];
    if (current >= 6) lowpass[current] -= 2*dcblock[current-6];
    if (current >= 12) lowpass[current] += dcblock[current-12];

    // High Pass filter
    highpass[current] = -lowpass[current];
    if (current >= 1) highpass[current] -= highpass[current-1];
    if (current >= 16) highpass[current] += 32*lowpass[current-16];
    if (current >= 32) highpass[current] += lowpass[current-32];

    // Derivative filter
    derivative[current] = highpass[current];
    if (current > 0) derivative[current] -= highpass[current-1];

    // Squaring
    squared[current] = derivative[current] * derivative[current];

    // Moving-Window Integration
    integral[current] = 0;
    for (i = 0; i < WINDOWSIZE; i++) {
        if (current >= (dataType)i)
            integral[current] += squared[current - i];
        else
            break;
    }
    if (i > 0) integral[current] /= (dataType)i;

    qrs = FALSE;

    if (integral[current] >= threshold_i1 || highpass[current] >= threshold_f1) {
        peak_i = integral[current];
        peak_f = highpass[current];
    }

    if ((integral[current] >= threshold_i1) && (highpass[current] >= threshold_f1)) {
        if (sample > lastQRS + FS/5) {
            if (sample <= lastQRS + (long unsigned int)(0.36*FS)) {
                currentSlope = 0;
                for (j = (current > 10 ? current - 10 : 0); j <= current; j++)
                    if (squared[j] > currentSlope) currentSlope = squared[j];

                if (currentSlope <= (dataType)(lastSlope/2)) {
                    qrs = FALSE;
                } else {
                    float learnRate = signalQuality.highNoiseMode ? 
                                    ADAPTIVE_LEARNING_RATE * 0.5 : ADAPTIVE_LEARNING_RATE;
                    
                    spk_i = learnRate*peak_i + (1-learnRate)*spk_i;
                    threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                    threshold_i2 = 0.5*threshold_i1;
                    spk_f = learnRate*peak_f + (1-learnRate)*spk_f;
                    threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                    threshold_f2 = 0.5*threshold_f1;
                    lastSlope = currentSlope;
                    qrs = TRUE;
                }
            } else {
                currentSlope = 0;
                for (j = (current > 10 ? current - 10 : 0); j <= current; j++)
                    if (squared[j] > currentSlope) currentSlope = squared[j];

                float learnRate = signalQuality.highNoiseMode ? 
                                ADAPTIVE_LEARNING_RATE * 0.5 : ADAPTIVE_LEARNING_RATE;
                
                spk_i = learnRate*peak_i + (1-learnRate)*spk_i;
                threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                threshold_i2 = 0.5*threshold_i1;
                spk_f = learnRate*peak_f + (1-learnRate)*spk_f;
                threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                threshold_f2 = 0.5*threshold_f1;
                lastSlope = currentSlope;
                qrs = TRUE;
            }
        } else {
            peak_i = integral[current];
            npk_i = ADAPTIVE_LEARNING_RATE*peak_i + (1-ADAPTIVE_LEARNING_RATE)*npk_i;
            threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
            threshold_i2 = 0.5*threshold_i1;
            peak_f = highpass[current];
            npk_f = ADAPTIVE_LEARNING_RATE*peak_f + (1-ADAPTIVE_LEARNING_RATE)*npk_f;
            threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
            threshold_f2 = 0.5*threshold_f1;
            qrs = FALSE;
            outputSignal[current] = qrs;
            if (sample > DELAY + BUFFSIZE) outputPT(outputSignal[0]);
            return;
        }
    }

    if (qrs) {
        amplitudeTracker.currentQRSAmplitude = peak_i;
        
        rravg1 = 0;
        for (i = 0; i < 7; i++) {
            rr1[i] = rr1[i+1];
            rravg1 += rr1[i];
        }
        rr1[7] = sample - lastQRS;
        lastQRS = sample;
        rravg1 += rr1[7];
        rravg1 *= 0.125;

        if ((rr1[7] >= rrlow) && (rr1[7] <= rrhigh)) {
            rravg2 = 0;
            for (i = 0; i < 7; i++) {
                rr2[i] = rr2[i+1];
                rravg2 += rr2[i];
            }
            rr2[7] = rr1[7];
            rravg2 += rr2[7];
            rravg2 *= 0.125;
            rrlow = 0.92*rravg2;
            rrhigh = 1.16*rravg2;
            rrmiss = 1.66*rravg2;
        }

        prevRegular = regular;
        if (rravg1 == rravg2) {
            regular = TRUE;
        } else {
            regular = FALSE;
            if (prevRegular) {
                threshold_i1 /= 2;
                threshold_f1 /= 2;
            }
        }
    } else {
        if ((sample - lastQRS > (long unsigned int)rrmiss) && (sample > lastQRS + FS/5)) {
            for (i = current - (sample - lastQRS) + FS/5; i < (long unsigned int)current; i++) {
                if ((integral[i] > threshold_i2) && (highpass[i] > threshold_f2)) {
                    currentSlope = 0;
                    for (j = (i > 10 ? i - 10 : 0); j <= i; j++)
                        if (squared[j] > currentSlope) currentSlope = squared[j];

                    if ((currentSlope < (dataType)(lastSlope/2)) && (i + sample) < lastQRS + 0.36*lastQRS) {
                        qrs = FALSE;
                    } else {
                        peak_i = integral[i];
                        peak_f = highpass[i];
                        amplitudeTracker.currentQRSAmplitude = peak_i;
                        
                        spk_i = 0.25*peak_i + 0.75*spk_i;
                        spk_f = 0.25*peak_f + 0.75*spk_f;
                        threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                        threshold_i2 = 0.5*threshold_i1;
                        lastSlope = currentSlope;
                        threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                        threshold_f2 = 0.5*threshold_f1;

                        rravg1 = 0;
                        for (j = 0; j < 7; j++) {
                            rr1[j] = rr1[j+1];
                            rravg1 += rr1[j];
                        }
                        rr1[7] = sample - (current - i) - lastQRS;
                        qrs = TRUE;
                        lastQRS = sample - (current - i);
                        rravg1 += rr1[7];
                        rravg1 *= 0.125;

                        if ((rr1[7] >= rrlow) && (rr1[7] <= rrhigh)) {
                            rravg2 = 0;
                            for (i = 0; i < 7; i++) {
                                rr2[i] = rr2[i+1];
                                rravg2 += rr2[i];
                            }
                            rr2[7] = rr1[7];
                            rravg2 += rr2[7];
                            rravg2 *= 0.125;
                            rrlow = 0.92*rravg2;
                            rrhigh = 1.16*rravg2;
                            rrmiss = 1.66*rravg2;
                        }

                        prevRegular = regular;
                        if (rravg1 == rravg2) {
                            regular = TRUE;
                        } else {
                            regular = FALSE;
                            if (prevRegular) {
                                threshold_i1 /= 2;
                                threshold_f1 /= 2;
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (!qrs) {
            if ((integral[current] >= threshold_i1) || (highpass[current] >= threshold_f1)) {
                peak_i = integral[current];
                npk_i = ADAPTIVE_LEARNING_RATE*peak_i + (1-ADAPTIVE_LEARNING_RATE)*npk_i;
                threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                threshold_i2 = 0.5*threshold_i1;
                peak_f = highpass[current];
                npk_f = ADAPTIVE_LEARNING_RATE*peak_f + (1-ADAPTIVE_LEARNING_RATE)*npk_f;
                threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                threshold_f2 = 0.5*threshold_f1;
            }
        }
    }

    outputSignal[current] = qrs;
    if (sample > DELAY + BUFFSIZE) outputPT(outputSignal[0]);
}

// ============ QRS DETECTION HANDLER ============
void handleQRSDetection() {
    if (qrsDetectedFlag) {
        qrsDetectedFlag = 0;
        rpeakMarkerReady = true;

        unsigned long currentTime = millis();
        
        if (!firstBeat && lastBeatTime > 0) {
            unsigned long rrInterval = currentTime - lastBeatTime;
            processRR(rrInterval, currentTime, amplitudeTracker.currentQRSAmplitude);
        } else {
            firstBeat = false;
        }
        
        lastBeatTime = currentTime;
        beatCount++;
    }
}

// ============ WAKEUP ISR ============
void wakeupPinISR() {
  wakeupTriggered = true;
  lastActivityTime = millis();
}

// ============ ECG ISR ============
void ecg_ISR() {
    ecg_rawSample = analogRead(ecgAnalogPin);
    ecg_inputBuffer[0] = (float32_t)ecg_rawSample;
    
    arm_fir_f32(&ecg_fir_highpass, ecg_inputBuffer, ecg_highpassOutput, 1);
    arm_fir_f32(&ecg_fir_notch_50hz, ecg_highpassOutput, ecg_notchOutput, 1);
    arm_fir_f32(&ecg_fir_lowpass, ecg_notchOutput, ecg_lowpassOutput, 1);


    ecg_filteredValue = (uint16_t)constrain(ecg_lowpassOutput[0], 0, 4095);
    
    // Push to UART buffer
    uart_push(ecg_filteredValue);
    
    // Feed to Pan-Tompkins
    currentECGSample = (dataType)ecg_filteredValue;
    newSampleReady = true;
}

// ============ PCG ISR ============
void pcg_ISR() {
    pcg_rawHeartSample = analogRead(heartMicPin);
    pcg_rawNoiseSample = analogRead(noiseMicPin);
    
    float32_t heartSignal = (pcg_rawHeartSample - 2048.0f) / 2048.0f;
    float32_t noiseReference = (pcg_rawNoiseSample - 2048.0f) / 2048.0f;
    
    if (validateNoiseReference(heartSignal, noiseReference)) {
        heartSignal = applyANC(heartSignal, noiseReference);
    }
    
    pcg_heartInputBuffer[0] = heartSignal;
    
    arm_fir_f32(&pcg_fir_lowpass, pcg_heartInputBuffer, pcg_lowpassOutput, 1);
    arm_fir_f32(&pcg_fir_notch_50hz, pcg_lowpassOutput, pcg_notchOutput, 1);
    
    heartSignal = pcg_notchOutput[0];
    heartSignal *= VOLUME_BOOST_FACTOR;
    heartSignal = constrain(heartSignal, -1.0f, 1.0f);
    
    pcg_filteredValue = (uint16_t)constrain((heartSignal + 1.0f) * 2048.0f, 0, 4095);
    analogWrite(pcgDacPin, pcg_filteredValue);
}

// ============ POWER MANAGEMENT ============
void enterSleepMode() {
  Serial.println("💤 STM32 entering SLEEP mode...");
  Serial.flush();
  Serial3.flush();
  
  ECG_Timer.detachInterrupt();
  PCG_Timer.detachInterrupt();
  
  uart_read_index = 0;
  uart_write_index = 0;
  
  systemActive = false;
  
  HAL_SuspendTick();
  HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
  HAL_ResumeTick();
  
  Serial.println("⏰ STM32 woke from sleep");
}

void wakeFromSleep() {
  ECG_Timer.attachInterruptInterval(ECG_SAMPLE_INTERVAL_US, ecg_ISR);
  PCG_Timer.attachInterruptInterval(PCG_SAMPLE_INTERVAL_US, pcg_ISR);
  
  lastActivityTime = millis();
  systemActive = true;
  wakeupTriggered = false;
  
  uart_read_index = 0;
  uart_write_index = 0;
  droppedSamples = 0;
}

// ============ SETUP ============
void setup() {
  Serial.begin(115200);
  Serial3.begin(115200);
  
  analogReadResolution(12);
  analogWriteResolution(12);
  
  pinMode(ecgAnalogPin, INPUT_ANALOG);
  pinMode(heartMicPin, INPUT_ANALOG);
  pinMode(noiseMicPin, INPUT_ANALOG);
  pinMode(pcgDacPin, OUTPUT);
  
  // Configure wake-up pin
  pinMode(WAKEUP_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(WAKEUP_PIN), wakeupPinISR, FALLING);
  
  lastActivityTime = millis();
  systemActive = true;
  
  // Initialize ECG filters
  arm_fir_init_f32(&ecg_fir_notch_50hz, ECG_NOTCH_TAPS, 
                   (float32_t *)ecg_notch_coeffs, ecg_notch_state, BLOCK_SIZE);
  arm_fir_init_f32(&ecg_fir_lowpass, ECG_LOWPASS_TAPS, 
                   (float32_t *)ecg_lowpass_coeffs, ecg_lowpass_state, BLOCK_SIZE);
  arm_fir_init_f32(&ecg_fir_highpass, 81,
                 (float32_t *)ecg_highpass_coeffs,ecg_highpass_state, BLOCK_SIZE);
  // Initialize PCG filters
  arm_fir_init_f32(&pcg_fir_notch_50hz, PCG_NOTCH_TAPS, 
                   (float32_t *)pcg_notch_coeffs, pcg_notch_state, BLOCK_SIZE);
  arm_fir_init_f32(&pcg_fir_lowpass, PCG_LOWPASS_TAPS, 
                   (float32_t *)pcg_lowpass_coeffs, pcg_lowpass_state, BLOCK_SIZE);
  
  // Initialize ANC coefficients
  for (int i = 0; i < ANC_FILTER_LENGTH; i++) {
    anc_coeffs[i] = 0.01f;
  }
  
  // Initialize Pan-Tompkins
  initPT();
  
  // Start timers
  if (!ECG_Timer.attachInterruptInterval(ECG_SAMPLE_INTERVAL_US, ecg_ISR)) {
    Serial.println("❌ ECG Timer FAILED");
    while(1);
  }
  
  if (!PCG_Timer.attachInterruptInterval(PCG_SAMPLE_INTERVAL_US, pcg_ISR)) {
    Serial.println("❌ PCG Timer FAILED");
    while(1);
  }
  
  delay(500);
 // Serial.println("\n=== STM32 ECG/PCG + Pan-Tompkins READY ===");
  //Serial.println("✅ 360Hz ECG with QRS detection");
  //Serial.println("✅ Heart rate calculation and transmission");
  //Serial.println("✅ 1000Hz PCG with ANC");
  //Serial.println("✅ Binary UART protocol at 115200 baud");
  //Serial.println("✅ 5-minute sleep/wake with ESP32");
  //Serial.println("🚀 System operational!");
}

// ============ MAIN LOOP ============
void loop() {
  // ============ HANDLE WAKE-UP ============
  if (wakeupTriggered && !systemActive) {
    wakeFromSleep();
  }
  
  // ============ PAN-TOMPKINS PROCESSING ============
  if (newSampleReady) {
    newSampleReady = false;
    panTompkins();
    handleQRSDetection();
  }
  
  // ============ UART TRANSMISSION ============
  static uint32_t lastUartTime = 0;
  
  if (micros() - lastUartTime >= 1000) {
    lastUartTime = micros();
    
    uint16_t buffered = uart_buffer_level();
    
    uint8_t maxBurst;
    if (buffered > 400) maxBurst = 30;
    else if (buffered > 256) maxBurst = 20;
    else if (buffered > 128) maxBurst = 15;
    else maxBurst = 10;
    
    uint16_t val;
    uint8_t sent = 0;
    uint8_t txBuffer[4];
    
    // Send ECG samples
    while (uart_pop(&val) && sent < maxBurst) {
      txBuffer[0] = 0xAA;  // ECG packet header
      txBuffer[1] = (val >> 8) & 0xFF;
      txBuffer[2] = val & 0xFF;
      txBuffer[3] = 0x55;  // ECG packet footer
      
      Serial3.write(txBuffer, 4);
      sent++;
    }
    
    // Send heart rate if updated
    if (bpmUpdateReady) {
      bpmUpdateReady = false;
      
      txBuffer[0] = 0xBB;  // BPM packet header
      txBuffer[1] = (transmitBPM >> 8) & 0xFF;
      txBuffer[2] = transmitBPM & 0xFF;
      txBuffer[3] = 0x66;  // BPM packet footer
      
      Serial3.write(txBuffer, 4);
    }

    // Send R-peak marker if detected
    if (rpeakMarkerReady) {
      rpeakMarkerReady = false;
      txBuffer[0] = 0xCC;
      txBuffer[1] = 0x00;
      txBuffer[2] = 0x01;
      txBuffer[3] = 0x77;
      Serial3.write(txBuffer, 4);
    }
  }
  
  // ============ MONITORING ============
  static uint32_t lastMonitorTime = 0;
  if (millis() - lastMonitorTime >= 2000) {
    lastMonitorTime = millis();
    
    Serial.print("📊 BPM:");
    Serial.print(transmitBPM);
    Serial.print(" | Beats:");
    Serial.print(validBeatCount);
    Serial.print(" | Q:");
    Serial.print(signalQuality.qualityScore);
    Serial.print("% | Buf:");
    Serial.print(uart_buffer_level());
    
    if (droppedSamples > 0) {
      Serial.print(" | Drop:");
      Serial.print(droppedSamples);
    }
    Serial.println();
  }
  
  // ============ POWER MANAGEMENT ============
  static uint32_t lastPowerCheck = 0;
  if (millis() - lastPowerCheck >= 5000) {
    lastPowerCheck = millis();
    
    if (systemActive && (millis() - lastActivityTime >= INACTIVITY_TIMEOUT)) {
      enterSleepMode();
    }
    
    if (wakeupTriggered && systemActive) {
      lastActivityTime = millis();
      wakeupTriggered = false;
    }
  }
}