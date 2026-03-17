// ESP32 ECG Display + TensorFlow Lite Micro Real AI Classification
// ✅ Professional scrolling waveform (right-to-left)
// ✅ Adjustable time scale with SELECT+UP/DOWN zoom
// ✅ UP/DOWN moves waveform vertically for centering
// ✅ Oscilloscope-style grid with time markers
// ✅ Binary UART protocol @ 360Hz
// ✅ Heart rate received from STM32 via UART
// ✅ R-Peak markers received from STM32 Pan-Tompkins [0xCC]
// ✅ REAL TFLite Micro inference — 4 classes: Normal, AF, AFL, PVC
// ✅ 10-second ECG collection → R-peak centered beat extraction → Multi-beat classification
// ✅ MODIFIED: X-axis expanded to show ~4 ECG waveforms instead of ~6
// ✅ NEW: Vertical position adjustment with UP/DOWN buttons
// ✅ FIXED: Honest signal quality — no fake stabilisation or random noise

#include <Arduino.h>
#include <TFT_eSPI.h>
#include <SPI.h>

// ===== TENSORFLOW LITE MICRO =====
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===== YOUR TRAINED MODEL =====
#include "ecg_model.h"

// ===== HARDWARE CONFIG =====
TFT_eSPI tft = TFT_eSPI();
#define TFT_WIDTH  320
#define TFT_HEIGHT 240

#define BUTTON_UP     13
#define BUTTON_DOWN   12
#define BUTTON_SELECT 14
#define BUTTON_EXIT   27

#define SerialSTM32 Serial2
#define STM32_WAKEUP_PIN 5

// ===== BINARY UART PROTOCOL =====
#define START_MARKER      0xAA
#define END_MARKER        0x55
#define BPM_START_MARKER  0xBB
#define BPM_END_MARKER    0x66
#define RPEAK_START_MARKER 0xCC
#define RPEAK_END_MARKER  0x77
#define SLEEP_CMD         0xFF
#define WAKE_CMD          0xFE

enum ParserState {
  WAIT_START,
  READ_HIGH,
  READ_LOW,
  READ_END,
  WAIT_BPM_HIGH,
  WAIT_BPM_LOW,
  WAIT_BPM_END,
  WAIT_RPEAK_HIGH,
  WAIT_RPEAK_LOW,
  WAIT_RPEAK_END
};

ParserState parserState = WAIT_START;
uint8_t highByte = 0;
uint8_t lowByte  = 0;
uint8_t bpmHighByte = 0;
uint8_t bpmLowByte  = 0;
uint8_t rpeakHighByte = 0;
uint8_t rpeakLowByte  = 0;

// Statistics for rate verification
uint32_t totalSamplesReceived = 0;
uint32_t invalidFrames   = 0;
uint32_t missedSamples   = 0;
uint32_t lastSampleTime  = 0;
uint32_t minInterval     = 999999;
uint32_t maxInterval     = 0;
uint32_t totalInterval   = 0;
uint32_t intervalCount   = 0;

// Rate measurement
uint32_t lastRateCheckTime = 0;
uint32_t samplesInLastSecond = 0;
float    measuredRate = 0.0;

// ===== TIMING CONFIG =====
struct TaskTiming {
  unsigned long lastRun;
  unsigned long interval;
};

TaskTiming tasks[] = {
  {0, 1},      // TASK_UART_READ
  {0, 2},      // TASK_PROCESS_HEART_RATE
  {0, 16},     // TASK_DISPLAY
  {0, 100},    // TASK_BUTTON
  {0, 7000},   // TASK_HR_UPDATE
  {0, 1000},   // TASK_SLEEP_CHECK
  {0, 1000},   // TASK_RATE_REPORT
  {0, 200}     // TASK_AI_COLLECTION
};

enum TaskID {
  TASK_UART_READ = 0,
  TASK_PROCESS_HEART_RATE,
  TASK_DISPLAY,
  TASK_BUTTON,
  TASK_HR_UPDATE,
  TASK_SLEEP_CHECK,
  TASK_RATE_REPORT,
  TASK_AI_COLLECTION
};

// ===== DISPLAY MODES =====
enum DisplayMode {
  DISPLAY_MAIN_MENU,
  DISPLAY_ECG_WAVEFORM,
  DISPLAY_AI_ANALYSIS,
  DISPLAY_AI_COLLECTING
};

DisplayMode currentDisplayMode = DISPLAY_MAIN_MENU;
bool displayNeedsFullRedraw = true;

// ===== OSCILLOSCOPE DISPLAY CONFIG =====
#define WAVEFORM_Y_START  25
#define WAVEFORM_Y_END    200
#define WAVEFORM_HEIGHT   (WAVEFORM_Y_END - WAVEFORM_Y_START)
#define WAVEFORM_CENTER   ((WAVEFORM_Y_START + WAVEFORM_Y_END) / 2)

const float ZOOM_LEVELS[] = {10.0, 5.0, 2.5, 1.0, 0.5};
const int NUM_ZOOM_LEVELS = 5;
int currentZoomLevel = 0;

int scrollX     = 0;
int sweepWidth  = 3;
float pixelsPerSample = 1.0;
bool newSampleAvailable = false;

// ============================================================
// Vertical offset is a VARIABLE, not #define
// ============================================================
int waveformVerticalOffset = 30;
#define VERTICAL_OFFSET_STEP   5
#define VERTICAL_OFFSET_MIN   -80
#define VERTICAL_OFFSET_MAX    80
// ============================================================

#define GRID_MAJOR      0x0187
#define GRID_MINOR      0x00C3
#define WAVEFORM_COLOR  0x07E0
#define HEADER_BG       0x0020

// ============================================================
// X-AXIS EXPANSION SYSTEM
// ============================================================
#define X_SCALE_FACTOR 1.5f

float fractionalScrollX  = 0.0f;
int   lastDrawnScrollX   = 0;
int   prevSampleY        = WAVEFORM_CENTER;
int   currentDrawY       = WAVEFORM_CENTER;
bool  hasPreviousSample  = false;
// ============================================================

// ===== SLEEP MODE =====
#define SLEEP_TIMEOUT_MS    150000
#define SLEEP_HOLD_TIME     2000
#define WAKE_PULSE_DURATION 100
unsigned long lastActivityTime = 0;
bool systemActive = true;

// ===== ECG DATA =====
struct ECGData {
  int   beatNumber    = 0;
  int   rrInterval    = 0;
  int   bpm           = 0;
  int   signalQuality = 0;
  float classProbs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

ECGData currentECG;

// ===== HEART RATE FROM STM32 =====
int bpm           = 0;
int signalQuality = 0;
int beatCount     = 0;

int lastWaveformY    = WAVEFORM_CENTER;
int mainMenuSelection = 0;

// ===== SIGNAL QUALITY TRACKING (REAL METRICS) =====
#define SQ_HISTORY_SIZE 32
int16_t sqRecentSamples[SQ_HISTORY_SIZE];
int     sqSampleIndex = 0;
bool    sqBufferFilled = false;
uint32_t sqClippedCount = 0;
uint32_t sqTotalCount   = 0;

// ===== 4 REAL CLASSES =====
const char* CLASS_NAMES[]  = {"Normal", "AF", "AFL", "PVC"};
const char* CLASS_ABBREV[] = {"Normal", "AF", "AFL", "PVC"};
const int   NUM_CLASSES    = 4;

// ===== TFLITE MICRO GLOBALS =====
namespace {
  tflite::ErrorReporter*    error_reporter  = nullptr;
  const tflite::Model*      tflModel        = nullptr;
  tflite::MicroInterpreter* interpreter     = nullptr;
  TfLiteTensor*             model_input     = nullptr;
  TfLiteTensor*             model_output    = nullptr;

  constexpr int kTensorArenaSize = 90 * 1024;
  uint8_t* tensor_arena = nullptr;
  bool     tfliteReady  = false;
}

// ===== ECG COLLECTION BUFFER FOR AI =====
#define AI_ECG_BUFFER_SIZE     3800
#define ECG_SAMPLE_RATE        360
#define BEAT_WINDOW            180
#define BEAT_SIZE              360
#define MAX_BEATS_TO_CLASSIFY  20
#define MIN_BEATS_FOR_RESULT   3
#define CONFIDENCE_THRESHOLD   0.50f

int16_t  aiEcgBuffer[AI_ECG_BUFFER_SIZE];
volatile int  aiEcgWriteIndex    = 0;
volatile bool aiBufferCollecting = false;

// ===== R-PEAK POSITIONS FROM STM32 =====
#define MAX_RPEAKS 30
int      rpeakPositions[MAX_RPEAKS];
volatile int rpeakCount = 0;

// Per-beat classification results
float beatResults[MAX_BEATS_TO_CLASSIFY][4];
int   classifiedBeatCount = 0;
int   validBeatCount_ai   = 0;
int   rejectedBeatCount   = 0;

// ===== AI ANALYSIS STATE =====
bool          aiCollectionInProgress  = false;
unsigned long aiCollectionStartTime   = 0;
const unsigned long AI_COLLECTION_DURATION = 10000;
int           aiCollectionProgress    = 0;
bool          aiResultsReady          = false;
bool          aiCollectionScreenDrawn = false;
bool          aiInferenceComplete     = false;
String        aiStatusMessage         = "";

int16_t currentECGSample = 0;

// ===== FUNCTION DECLARATIONS =====
void task_UART_Read();
void task_Display();
void task_Button();
void task_SleepCheck();
void task_RateReport();
void task_AICollection();

void processBinaryByte(uint8_t byte);
void processECGSample(int16_t sample);
void processBPMData(uint16_t bpmValue);
void processRPeakMarker();

void drawMainMenu();
void drawECGWaveform();
void drawOscilloscopeGrid();
void drawTimeScale();
void drawAIAnalysis();
void drawAICollectionScreen();
void updateAICollectionProgress();

void handleButtons();
void enterSleepMode();
void wakeFromSleep();
void sendWakePulseToSTM32();

void updateSignalQuality();
void startAICollection();
void updateAICollection();

bool initTFLite();
bool extractAndNormalizeBeat(int16_t* buffer, int bufferLen,
                             int rpeakPos, float* beatOut);
bool classifySingleBeat(float* normalizedBeat, float* probsOut);
void runFullInference();
void aggregateResults();

// ===== SETUP =====
void setup() {
  Serial.begin(115200);

  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);

  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(80, 100);
  tft.print("Starting...");

  pinMode(BUTTON_UP,     INPUT_PULLUP);
  pinMode(BUTTON_DOWN,   INPUT_PULLUP);
  pinMode(BUTTON_SELECT, INPUT_PULLUP);
  pinMode(BUTTON_EXIT,   INPUT_PULLUP);

  pinMode(STM32_WAKEUP_PIN, OUTPUT);
  digitalWrite(STM32_WAKEUP_PIN, HIGH);

  SerialSTM32.begin(115200, SERIAL_8N1, 16, 17);

  lastActivityTime  = millis();
  lastRateCheckTime = millis();

  // Initialize signal quality tracking
  memset(sqRecentSamples, 0, sizeof(sqRecentSamples));
  sqSampleIndex  = 0;
  sqBufferFilled = false;
  sqClippedCount = 0;
  sqTotalCount   = 0;

  // Initialize TFLite
  tft.fillScreen(TFT_BLACK);
  tft.setCursor(40, 100);
  tft.setTextSize(2);
  tft.setTextColor(TFT_YELLOW);
  tft.print("Loading AI Model...");

  bool modelLoaded = initTFLite();

  tft.fillScreen(TFT_BLACK);
  if (modelLoaded) {
    tft.setTextColor(TFT_GREEN);
    tft.setCursor(40, 100);
    tft.print("AI Model Ready!");
    Serial.println("TFLite model loaded");
  } else {
    tft.setTextColor(TFT_RED);
    tft.setCursor(30, 100);
    tft.print("AI Model FAILED!");
    Serial.println("TFLite model failed");
  }
  delay(800);

  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);
  tft.setCursor(60, 100);
  tft.setTextSize(2);
  tft.print("System Ready");
  delay(500);

  currentDisplayMode    = DISPLAY_MAIN_MENU;
  displayNeedsFullRedraw = true;
}

// ===== TFLITE INITIALIZATION =====
bool initTFLite() {
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  if (tensor_arena == nullptr) {
    Serial.println("Failed to allocate tensor arena");
    return false;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  tflModel = tflite::GetModel(ecg_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema %d != %d\n",
                  tflModel->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }

  model_input  = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.printf("Input: shape=[%d,%d,%d] type=%d scale=%.6f zp=%d\n",
    model_input->dims->data[0], model_input->dims->data[1],
    model_input->dims->size > 2 ? model_input->dims->data[2] : 0,
    model_input->type, model_input->params.scale,
    model_input->params.zero_point);

  Serial.printf("Output: shape=[%d,%d] type=%d scale=%.6f zp=%d\n",
    model_output->dims->data[0], model_output->dims->data[1],
    model_output->type, model_output->params.scale,
    model_output->params.zero_point);

  Serial.printf("TFLite ready! Arena: %d bytes\n",
                interpreter->arena_used_bytes());

  tfliteReady = true;
  return true;
}

// ===== BINARY UART PARSING =====
void processBinaryByte(uint8_t byte) {
  switch (parserState) {
    case WAIT_START:
      if      (byte == START_MARKER)       parserState = READ_HIGH;
      else if (byte == BPM_START_MARKER)   parserState = WAIT_BPM_HIGH;
      else if (byte == RPEAK_START_MARKER) parserState = WAIT_RPEAK_HIGH;
      break;

    case READ_HIGH:
      highByte    = byte;
      parserState = READ_LOW;
      break;

    case READ_LOW:
      lowByte     = byte;
      parserState = READ_END;
      break;

    case READ_END:
      if (byte == END_MARKER) {
        uint16_t ecgValue = (highByte << 8) | lowByte;

        uint32_t now = micros();
        if (lastSampleTime > 0) {
          uint32_t interval = now - lastSampleTime;
          if (interval < minInterval) minInterval = interval;
          if (interval > maxInterval) maxInterval = interval;
          totalInterval += interval;
          intervalCount++;
          if (interval > 4000) missedSamples++;
        }
        lastSampleTime = now;

        processECGSample(ecgValue);
        totalSamplesReceived++;
        samplesInLastSecond++;
        lastActivityTime = millis();
      } else {
        invalidFrames++;
      }
      parserState = WAIT_START;
      break;

    case WAIT_BPM_HIGH:
      bpmHighByte = byte;
      parserState = WAIT_BPM_LOW;
      break;

    case WAIT_BPM_LOW:
      bpmLowByte  = byte;
      parserState = WAIT_BPM_END;
      break;

    case WAIT_BPM_END:
      if (byte == BPM_END_MARKER) {
        uint16_t bpmValue = (bpmHighByte << 8) | bpmLowByte;
        processBPMData(bpmValue);
        lastActivityTime = millis();
      } else {
        invalidFrames++;
      }
      parserState = WAIT_START;
      break;

    case WAIT_RPEAK_HIGH:
      rpeakHighByte = byte;
      parserState   = WAIT_RPEAK_LOW;
      break;

    case WAIT_RPEAK_LOW:
      rpeakLowByte  = byte;
      parserState   = WAIT_RPEAK_END;
      break;

    case WAIT_RPEAK_END:
      if (byte == RPEAK_END_MARKER) {
        processRPeakMarker();
        lastActivityTime = millis();
      } else {
        invalidFrames++;
      }
      parserState = WAIT_START;
      break;
  }
}

void processRPeakMarker() {
  if (aiBufferCollecting && rpeakCount < MAX_RPEAKS) {
    int rpeakPos = aiEcgWriteIndex - 25;
    if (rpeakPos >= BEAT_WINDOW &&
        rpeakPos < AI_ECG_BUFFER_SIZE - BEAT_WINDOW) {
      rpeakPositions[rpeakCount] = rpeakPos;
      rpeakCount++;
      Serial.printf("R-peak #%d at buffer pos %d\n",
                    rpeakCount, rpeakPos);
    }
  }
}

void sendWakePulseToSTM32() {
  pinMode(STM32_WAKEUP_PIN, OUTPUT);
  digitalWrite(STM32_WAKEUP_PIN, LOW);
  delay(WAKE_PULSE_DURATION);
  digitalWrite(STM32_WAKEUP_PIN, HIGH);
}

void processBPMData(uint16_t bpmValue) {
  bpm = bpmValue;
  currentECG.bpm = bpm;
  beatCount++;
  currentECG.beatNumber = beatCount;
  // Signal quality is updated every sample in processECGSample,
  // but we can also trigger a recalc here
  updateSignalQuality();
}

// ======================================================================
// HONEST SIGNAL QUALITY — Based on REAL measurable metrics
// No random noise, no artificial floors/ceilings, no fake stabilisation
// ======================================================================
void updateSignalQuality() {
  // ─── Metric 1: Data reception rate (0-100) ───
  // Are we actually receiving samples at the expected ~360 Hz?
  int rateScore;
  if (measuredRate >= 340.0f && measuredRate <= 380.0f) {
    rateScore = 100;    // Within ±20 Hz of 360 — excellent
  } else if (measuredRate >= 300.0f && measuredRate <= 420.0f) {
    rateScore = 70;     // Acceptable deviation
  } else if (measuredRate >= 100.0f) {
    rateScore = 30;     // Significant data loss but something coming in
  } else if (measuredRate > 0.0f) {
    rateScore = 10;     // Very poor reception
  } else {
    rateScore = 0;      // No data at all
  }

  // ─── Metric 2: ADC range validity (0-100) ───
  // Check if the current sample is within usable ADC range
  // 12-bit ADC: 0-4095, railed signals indicate electrode problems
  int rangeScore;
  if (currentECGSample > 100 && currentECGSample < 3995) {
    rangeScore = 100;   // Clean — well within ADC range
  } else if (currentECGSample > 20 && currentECGSample < 4075) {
    rangeScore = 50;    // Near rails — possible saturation
  } else if (currentECGSample > 5 && currentECGSample < 4090) {
    rangeScore = 20;    // Almost railed — likely electrode issue
  } else {
    rangeScore = 0;     // Fully railed or no signal
  }

  // ─── Metric 3: Signal variance check (0-100) ───
  // A flat line (zero variance) means no real ECG signal
  // Use the rolling buffer of recent samples
  int varianceScore = 0;
  if (sqBufferFilled || sqSampleIndex >= 8) {
    int count = sqBufferFilled ? SQ_HISTORY_SIZE : sqSampleIndex;
    float mean = 0.0f;
    for (int i = 0; i < count; i++) {
      mean += sqRecentSamples[i];
    }
    mean /= count;

    float variance = 0.0f;
    for (int i = 0; i < count; i++) {
      float diff = sqRecentSamples[i] - mean;
      variance += diff * diff;
    }
    variance /= count;

    // Typical ECG has noticeable variance; flat line has ~0
    // Very high variance might indicate noise/movement artifact
    if (variance > 500.0f && variance < 500000.0f) {
      varianceScore = 100;   // Good dynamic range
    } else if (variance > 100.0f && variance < 1000000.0f) {
      varianceScore = 70;    // Acceptable
    } else if (variance > 10.0f) {
      varianceScore = 30;    // Very low signal or excessive noise
    } else {
      varianceScore = 0;     // Flat line — no real signal
    }
  } else {
    // Not enough samples yet to judge
    varianceScore = 0;
  }

  // ─── Metric 4: BPM physiological plausibility (0-100) ───
  // Is the detected heart rate within human-possible range?
  int bpmScore;
  if (bpm >= 40 && bpm <= 180) {
    bpmScore = 100;     // Normal physiological range
  } else if (bpm >= 30 && bpm <= 220) {
    bpmScore = 60;      // Edge of plausible (athlete/exercise)
  } else if (bpm > 0 && bpm < 300) {
    bpmScore = 20;      // Likely artifact or miscounting
  } else {
    bpmScore = 0;       // No heartbeat detected or absurd value
  }

  // ─── Metric 5: Frame error rate (0-100) ───
  // How many UART frames had protocol errors?
  int frameScore;
  if (totalSamplesReceived > 0) {
    float errorRate = (float)invalidFrames / (float)(totalSamplesReceived + invalidFrames);
    if (errorRate < 0.001f) {
      frameScore = 100;     // < 0.1% errors — excellent
    } else if (errorRate < 0.01f) {
      frameScore = 80;      // < 1% errors — good
    } else if (errorRate < 0.05f) {
      frameScore = 50;      // < 5% errors — degraded
    } else if (errorRate < 0.10f) {
      frameScore = 25;      // < 10% errors — poor
    } else {
      frameScore = 0;       // > 10% errors — very poor
    }
  } else {
    frameScore = 0;         // No data received
  }

  // ─── Combine all metrics ───
  // Weighted average: reception and variance are most important
  // rateScore:     25% — is data arriving?
  // rangeScore:    20% — is ADC in valid range?
  // varianceScore: 25% — is there actual signal variation?
  // bpmScore:      15% — is heart rate plausible?
  // frameScore:    15% — is communication clean?
  int combined = (rateScore     * 25 +
                  rangeScore    * 20 +
                  varianceScore * 25 +
                  bpmScore      * 15 +
                  frameScore    * 15) / 100;

  // Full 0-100 range — bad signal SHOULD show bad quality
  signalQuality = constrain(combined, 0, 100);
  currentECG.signalQuality = signalQuality;
}

// ===== SLEEP MODE =====
void enterSleepMode() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(3);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(50, 100);
  tft.print("SLEEP MODE");
  tft.setTextSize(2);
  tft.setCursor(30, 140);
  tft.print("Press EXIT to wake");

  delay(1000);
  esp_sleep_enable_ext0_wakeup((gpio_num_t)BUTTON_EXIT, 0);
  esp_light_sleep_start();
  wakeFromSleep();
}

void wakeFromSleep() {
  sendWakePulseToSTM32();
  delay(200);
  lastActivityTime = millis();
  systemActive     = true;
  displayNeedsFullRedraw = true;
  totalSamplesReceived   = 0;
  invalidFrames          = 0;
  missedSamples          = 0;
  lastSampleTime         = 0;
  minInterval            = 999999;
  maxInterval            = 0;

  // Reset signal quality tracking on wake
  memset(sqRecentSamples, 0, sizeof(sqRecentSamples));
  sqSampleIndex  = 0;
  sqBufferFilled = false;
  sqClippedCount = 0;
  sqTotalCount   = 0;
  signalQuality  = 0;
  currentECG.signalQuality = 0;

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_GREEN);
  tft.setCursor(80, 110);
  tft.print("SYSTEM ACTIVE");
  delay(500);
}

// ======================================================================
// REAL AI INFERENCE FUNCTIONS
// ======================================================================

bool extractAndNormalizeBeat(int16_t* buffer, int bufferLen,
                             int rpeakPos, float* beatOut) {
  int start = rpeakPos - BEAT_WINDOW;
  int end   = rpeakPos + BEAT_WINDOW;

  if (start < 0 || end > bufferLen) return false;

  float rawBeat[BEAT_SIZE];
  for (int i = 0; i < BEAT_SIZE; i++)
    rawBeat[i] = (float)buffer[start + i];

  float minVal = rawBeat[0], maxVal = rawBeat[0];
  for (int i = 1; i < BEAT_SIZE; i++) {
    if (rawBeat[i] < minVal) minVal = rawBeat[i];
    if (rawBeat[i] > maxVal) maxVal = rawBeat[i];
  }

  float range = maxVal - minVal;
  if (range < 10.0f) {
    Serial.println("  Beat rejected: flat signal");
    return false;
  }

  int clipCount = 0;
  for (int i = 0; i < BEAT_SIZE; i++) {
    if (rawBeat[i] <= 5.0f || rawBeat[i] >= 4090.0f) clipCount++;
  }
  if (clipCount > BEAT_SIZE * 0.1) {
    Serial.println("  Beat rejected: clipped signal");
    return false;
  }

  float mean = 0.0f;
  for (int i = 0; i < BEAT_SIZE; i++) mean += rawBeat[i];
  mean /= BEAT_SIZE;
  for (int i = 0; i < BEAT_SIZE; i++) rawBeat[i] -= mean;

  minVal = rawBeat[0]; maxVal = rawBeat[0];
  for (int i = 1; i < BEAT_SIZE; i++) {
    if (rawBeat[i] < minVal) minVal = rawBeat[i];
    if (rawBeat[i] > maxVal) maxVal = rawBeat[i];
  }
  range = maxVal - minVal;
  if (range < 1e-6f) return false;

  for (int i = 0; i < BEAT_SIZE; i++)
    beatOut[i] = 2.0f * (rawBeat[i] - minVal) / range - 1.0f;

  return true;
}

bool classifySingleBeat(float* normalizedBeat, float* probsOut) {
  if (!tfliteReady || interpreter == nullptr) return false;

  float input_scale      = model_input->params.scale;
  int   input_zero_point = model_input->params.zero_point;

  int8_t* input_data = model_input->data.int8;
  for (int i = 0; i < BEAT_SIZE; i++) {
    float val = normalizedBeat[i];
    if (val < -1.0f) val = -1.0f;
    if (val >  1.0f) val =  1.0f;

    int32_t quantized = (int32_t)roundf(val / input_scale) + input_zero_point;
    if (quantized < -128) quantized = -128;
    if (quantized >  127) quantized =  127;
    input_data[i] = (int8_t)quantized;
  }

  unsigned long inferStart   = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inferTime    = micros() - inferStart;

  if (invoke_status != kTfLiteOk) {
    Serial.println("  Inference failed");
    return false;
  }

  float   output_scale      = model_output->params.scale;
  int     output_zero_point = model_output->params.zero_point;
  int8_t* output_data       = model_output->data.int8;

  float sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    probsOut[i] = (output_data[i] - output_zero_point) * output_scale;
    if (probsOut[i] < 0.0f) probsOut[i] = 0.0f;
    sum += probsOut[i];
  }

  if (sum > 0.0f) {
    for (int i = 0; i < NUM_CLASSES; i++)
      probsOut[i] = (probsOut[i] / sum) * 100.0f;
  }

  Serial.printf("  %.1fms | N=%.1f%% AF=%.1f%% AFL=%.1f%% PVC=%.1f%%\n",
    inferTime / 1000.0f,
    probsOut[0], probsOut[1], probsOut[2], probsOut[3]);

  return true;
}

void runFullInference() {
  Serial.println("\nStarting inference pipeline...");
  Serial.printf("Buffer: %d samples (%.1fs), R-peaks from STM32: %d\n",
    aiEcgWriteIndex, aiEcgWriteIndex / (float)ECG_SAMPLE_RATE, rpeakCount);

  if (!tfliteReady) {
    Serial.println("TFLite not initialized!");
    aiStatusMessage = "AI Model not loaded";
    for (int i = 0; i < NUM_CLASSES; i++) currentECG.classProbs[i] = 0.0f;
    aiInferenceComplete = true;
    return;
  }

  if (rpeakCount < 1) {
    Serial.println("No R-peaks received from STM32!");
    aiStatusMessage = "No heartbeats detected";
    for (int i = 0; i < NUM_CLASSES; i++) currentECG.classProbs[i] = 0.0f;
    aiInferenceComplete = true;
    return;
  }

  Serial.printf("Step 1: Classifying %d beats from STM32 R-peaks...\n",
                rpeakCount);

  classifiedBeatCount = 0;
  validBeatCount_ai   = 0;
  rejectedBeatCount   = 0;

  float normalizedBeat[BEAT_SIZE];

  for (int b = 0;
       b < rpeakCount && classifiedBeatCount < MAX_BEATS_TO_CLASSIFY;
       b++) {
    Serial.printf("  Beat %d/%d (R-peak at sample %d):\n",
                  b + 1, rpeakCount, rpeakPositions[b]);

    if (!extractAndNormalizeBeat(aiEcgBuffer, aiEcgWriteIndex,
                                 rpeakPositions[b], normalizedBeat)) {
      rejectedBeatCount++;
      continue;
    }

    float probs[NUM_CLASSES];
    if (classifySingleBeat(normalizedBeat, probs)) {
      for (int c = 0; c < NUM_CLASSES; c++)
        beatResults[classifiedBeatCount][c] = probs[c];
      classifiedBeatCount++;
      validBeatCount_ai++;
    } else {
      rejectedBeatCount++;
    }
  }

  Serial.printf("Step 2: Aggregating (%d valid, %d rejected)\n",
                validBeatCount_ai, rejectedBeatCount);
  aggregateResults();

  aiInferenceComplete = true;
  Serial.println("Inference complete!");
}

void aggregateResults() {
  if (classifiedBeatCount < MIN_BEATS_FOR_RESULT) {
    if (classifiedBeatCount > 0)
      aiStatusMessage = "Low confidence (few beats)";
    else {
      aiStatusMessage = "Could not classify any beats";
      for (int i = 0; i < NUM_CLASSES; i++)
        currentECG.classProbs[i] = 0.0f;
      return;
    }
  } else {
    aiStatusMessage = "";
  }

  float avgProbs[NUM_CLASSES] = {0.0f};
  for (int b = 0; b < classifiedBeatCount; b++)
    for (int c = 0; c < NUM_CLASSES; c++)
      avgProbs[c] += beatResults[b][c];

  for (int c = 0; c < NUM_CLASSES; c++)
    avgProbs[c] /= classifiedBeatCount;

  float total = 0.0f;
  for (int c = 0; c < NUM_CLASSES; c++) total += avgProbs[c];
  if (total > 0.0f)
    for (int c = 0; c < NUM_CLASSES; c++)
      avgProbs[c] = (avgProbs[c] / total) * 100.0f;

  for (int c = 0; c < NUM_CLASSES; c++)
    currentECG.classProbs[c] = avgProbs[c];

  int   dominantClass = 0;
  float maxProb       = avgProbs[0];
  for (int c = 1; c < NUM_CLASSES; c++) {
    if (avgProbs[c] > maxProb) {
      maxProb       = avgProbs[c];
      dominantClass = c;
    }
  }

  if (maxProb < CONFIDENCE_THRESHOLD * 100.0f)
    aiStatusMessage = "Inconclusive - retry";

  int pvcBeats = 0;
  for (int b = 0; b < classifiedBeatCount; b++)
    if (beatResults[b][3] > 50.0f) pvcBeats++;

  Serial.println("\n=== FINAL RESULTS ===");
  Serial.printf("  Beats: %d analyzed, %d rejected\n",
                classifiedBeatCount, rejectedBeatCount);
  for (int c = 0; c < NUM_CLASSES; c++)
    Serial.printf("  %s: %.1f%%\n",
                  CLASS_NAMES[c], currentECG.classProbs[c]);
  Serial.printf("  Dominant: %s (%.1f%%)\n",
                CLASS_NAMES[dominantClass], maxProb);
  if (pvcBeats > 0)
    Serial.printf("  PVC Burden: %.1f%%\n",
                  (float)pvcBeats / classifiedBeatCount * 100.0f);
  Serial.println("=====================");
}

// ===== AI COLLECTION =====
void startAICollection() {
  aiEcgWriteIndex     = 0;
  aiBufferCollecting  = true;
  rpeakCount          = 0;
  classifiedBeatCount = 0;
  validBeatCount_ai   = 0;
  rejectedBeatCount   = 0;
  aiInferenceComplete = false;
  aiStatusMessage     = "";

  for (int i = 0; i < NUM_CLASSES; i++)
    currentECG.classProbs[i] = 0.0f;

  aiCollectionInProgress  = true;
  aiCollectionStartTime   = millis();
  aiCollectionProgress    = 0;
  aiResultsReady          = false;
  aiCollectionScreenDrawn = false;

  currentDisplayMode     = DISPLAY_AI_COLLECTING;
  displayNeedsFullRedraw = true;

  Serial.println("AI collection started (10s)");
}

void updateAICollection() {
  if (!aiCollectionInProgress) return;

  unsigned long elapsed = millis() - aiCollectionStartTime;
  if (elapsed >= AI_COLLECTION_DURATION) {
    aiBufferCollecting     = false;
    aiCollectionInProgress = false;

    Serial.printf("Done: %d samples, %d R-peaks from STM32\n",
                  aiEcgWriteIndex, rpeakCount);

    tft.fillRect(10, 170, 300, 50, TFT_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(TFT_YELLOW);
    tft.setCursor(60, 180);
    tft.print("Analyzing...");

    runFullInference();

    aiResultsReady         = true;
    currentDisplayMode     = DISPLAY_AI_ANALYSIS;
    displayNeedsFullRedraw = true;
  } else {
    int newProgress = (elapsed * 100) / AI_COLLECTION_DURATION;
    if (abs(newProgress - aiCollectionProgress) >= 1) {
      aiCollectionProgress = newProgress;
      updateAICollectionProgress();
    }
  }
}

// ===== TASK FUNCTIONS =====
void task_UART_Read() {
  while (SerialSTM32.available()) {
    uint8_t byte = SerialSTM32.read();
    processBinaryByte(byte);
  }
}

void task_RateReport() {
  uint32_t now = millis();
  if (now - lastRateCheckTime >= 1000) {
    measuredRate         = samplesInLastSecond;
    samplesInLastSecond  = 0;
    minInterval          = 999999;
    maxInterval          = 0;
    totalInterval        = 0;
    intervalCount        = 0;
    lastRateCheckTime    = now;
  }
}

void task_ProcessHeartRate() {}

void processECGSample(int16_t sample) {
  currentECGSample   = sample;
  newSampleAvailable = true;

  // Track recent samples for signal quality variance calculation
  sqRecentSamples[sqSampleIndex] = sample;
  sqSampleIndex++;
  if (sqSampleIndex >= SQ_HISTORY_SIZE) {
    sqSampleIndex  = 0;
    sqBufferFilled = true;
  }

  // Track clipping for signal quality
  sqTotalCount++;
  if (sample <= 5 || sample >= 4090) {
    sqClippedCount++;
  }

  // Update signal quality periodically (every 36 samples = ~10Hz at 360Hz)
  if (sqTotalCount % 36 == 0) {
    updateSignalQuality();
  }

  if (aiBufferCollecting && aiEcgWriteIndex < AI_ECG_BUFFER_SIZE) {
    aiEcgBuffer[aiEcgWriteIndex] = sample;
    aiEcgWriteIndex++;
  }
}

void task_AICollection() {
  updateAICollection();
}

// ===== DISPLAY FUNCTIONS =====
void drawOscilloscopeGrid() {
  float timeWindow         = ZOOM_LEVELS[currentZoomLevel];
  float effectiveTimeWindow = timeWindow / X_SCALE_FACTOR;
  float pixelsPerSecond     = TFT_WIDTH / effectiveTimeWindow;
  float majorInterval       = (effectiveTimeWindow <= 2.5) ? 0.2 : 1.0;
  int   majorSpacing        = (int)(pixelsPerSecond * majorInterval);
  int   minorSpacing        = (effectiveTimeWindow <= 1.0)
                                ? (int)(pixelsPerSecond * 0.04)
                                : (int)(pixelsPerSecond * 0.2);

  if (majorSpacing < 5) majorSpacing = 5;
  if (minorSpacing < 2) minorSpacing = 2;

  for (int x = 0; x < TFT_WIDTH; x++) {
    if (majorSpacing > 0 && x % majorSpacing == 0) {
      for (int y = WAVEFORM_Y_START; y <= WAVEFORM_Y_END; y += 2)
        tft.drawPixel(x, y, GRID_MAJOR);
    } else if (minorSpacing > 0 && x % minorSpacing == 0) {
      for (int y = WAVEFORM_Y_START; y <= WAVEFORM_Y_END; y += 4)
        tft.drawPixel(x, y, GRID_MINOR);
    }
  }

  for (int y = WAVEFORM_Y_START; y <= WAVEFORM_Y_END; y++) {
    int relY = y - WAVEFORM_Y_START;
    if (relY % 40 == 0) {
      for (int x = 0; x < TFT_WIDTH; x += 2)
        tft.drawPixel(x, y, GRID_MAJOR);
    } else if (relY % 8 == 0) {
      for (int x = 0; x < TFT_WIDTH; x += 4)
        tft.drawPixel(x, y, GRID_MINOR);
    }
  }
}

void drawTimeScale() {
  float timeWindow          = ZOOM_LEVELS[currentZoomLevel];
  float effectiveTimeWindow = timeWindow / X_SCALE_FACTOR;

  tft.setTextSize(1);
  tft.setTextColor(TFT_CYAN, TFT_BLACK);

  int   numMarkers = (effectiveTimeWindow <= 1.0) ? 5
                   : ((effectiveTimeWindow <= 2.5) ? 6 : 11);
  float timeStep   = effectiveTimeWindow / (numMarkers - 1);

  for (int i = 0; i < numMarkers; i++) {
    int   x    = (i * TFT_WIDTH) / (numMarkers - 1);
    float time = i * timeStep;
    tft.drawFastVLine(x, WAVEFORM_Y_END + 2, 3, TFT_CYAN);

    char timeStr[8];
    if (effectiveTimeWindow <= 1.0)
      sprintf(timeStr, "%dms", (int)(time * 1000));
    else
      sprintf(timeStr, "%.1fs", time);

    int textWidth = strlen(timeStr) * 6;
    tft.setCursor(x - textWidth / 2, WAVEFORM_Y_END + 6);
    tft.print(timeStr);
  }
}

void task_Display() {
  switch (currentDisplayMode) {
    case DISPLAY_MAIN_MENU:
      if (displayNeedsFullRedraw) {
        drawMainMenu();
        displayNeedsFullRedraw = false;
      }
      break;

    case DISPLAY_ECG_WAVEFORM:
      if (displayNeedsFullRedraw) {
        tft.fillScreen(TFT_BLACK);
        drawOscilloscopeGrid();
        drawTimeScale();
        scrollX            = 0;
        fractionalScrollX  = 0.0f;
        lastDrawnScrollX   = 0;
        lastWaveformY      = WAVEFORM_CENTER;
        prevSampleY        = WAVEFORM_CENTER;
        hasPreviousSample  = false;
        displayNeedsFullRedraw = false;
      }
      drawECGWaveform();
      break;

    case DISPLAY_AI_COLLECTING:
      if (displayNeedsFullRedraw || !aiCollectionScreenDrawn) {
        drawAICollectionScreen();
        aiCollectionScreenDrawn = true;
        displayNeedsFullRedraw  = false;
      }
      break;

    case DISPLAY_AI_ANALYSIS:
      if (displayNeedsFullRedraw) {
        drawAIAnalysis();
        displayNeedsFullRedraw = false;
      }
      break;
  }
}

void drawMainMenu() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(3);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(80, 10);
  tft.print("MAIN MENU");

  for (int i = 0; i < 2; i++) {
    int x      = 60 + i * 120;
    int y      = 80;
    int width  = 100;
    int height = 50;

    if (i == mainMenuSelection) {
      tft.fillRect(x, y, width, height, TFT_YELLOW);
      tft.setTextColor(TFT_BLACK);
    } else {
      uint16_t color = (i == 0) ? TFT_BLUE : TFT_GREEN;
      tft.fillRect(x, y, width, height, color);
      tft.setTextColor(TFT_WHITE);
    }
    tft.setCursor(x + 25, y + 15);
    tft.print(i == 0 ? "ECG" : "AI");
  }

  tft.setTextSize(1);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(10, 150);
  tft.print("BPM: ");  tft.print(bpm);
  tft.print(" | Quality: "); tft.print(signalQuality);
  tft.print("%");

  tft.setCursor(10, 165);
  tft.print("Rate: "); tft.print(measuredRate, 1);
  tft.print(" Hz | Beats: "); tft.print(beatCount);

  tft.setCursor(10, 175);
  if (tfliteReady) {
    tft.setTextColor(TFT_GREEN);
    tft.print("AI Model: Ready");
  } else {
    tft.setTextColor(TFT_RED);
    tft.print("AI Model: NOT LOADED");
  }

  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(20, 190);
  tft.print("UP/DOWN: Select");
  tft.setCursor(20, 210);
  tft.print("SELECT: Enter");
  tft.setCursor(20, 230);
  tft.print("EXIT: Back");
}

// ============================================================
// drawECGWaveform() — uses variable waveformVerticalOffset
// ============================================================
void drawECGWaveform() {
  if (!newSampleAvailable) return;
  newSampleAvailable = false;

  float scale       = 0.06;
  int   scaledValue = (int)((currentECGSample - 2048) * scale);
  int   y           = WAVEFORM_CENTER - scaledValue + waveformVerticalOffset;
  y = constrain(y, WAVEFORM_Y_START, WAVEFORM_Y_END);

  int prevPixelX = (int)fractionalScrollX % TFT_WIDTH;
  int prevY      = prevSampleY;

  fractionalScrollX += X_SCALE_FACTOR;

  if (fractionalScrollX >= (float)TFT_WIDTH)
    fractionalScrollX -= (float)TFT_WIDTH;

  int newPixelX     = (int)fractionalScrollX % TFT_WIDTH;
  int pixelsCrossed;

  if (newPixelX >= prevPixelX)
    pixelsCrossed = newPixelX - prevPixelX;
  else
    pixelsCrossed = (TFT_WIDTH - prevPixelX) + newPixelX;

  // Clear ahead
  int clearAhead = 20;
  for (int i = 1; i <= clearAhead; i++) {
    int clearX = (newPixelX + i) % TFT_WIDTH;
    tft.drawFastVLine(clearX, WAVEFORM_Y_START,
                      WAVEFORM_HEIGHT, TFT_BLACK);
  }

  // Redraw grid in cleared region
  for (int i = 1; i <= clearAhead; i++) {
    int clearX = (newPixelX + i) % TFT_WIDTH;

    float timeWindow          = ZOOM_LEVELS[currentZoomLevel];
    float effectiveTimeWindow = timeWindow / X_SCALE_FACTOR;
    float pixelsPerSecond     = TFT_WIDTH / effectiveTimeWindow;
    float majorInterval_f     = (effectiveTimeWindow <= 2.5) ? 0.2 : 1.0;
    int   majorSpacing        = (int)(pixelsPerSecond * majorInterval_f);
    int   minorSpacing        = (effectiveTimeWindow <= 1.0)
                                  ? (int)(pixelsPerSecond * 0.04)
                                  : (int)(pixelsPerSecond * 0.2);

    if (majorSpacing < 5) majorSpacing = 5;
    if (minorSpacing < 2) minorSpacing = 2;

    for (int gy = WAVEFORM_Y_START; gy <= WAVEFORM_Y_END; gy++) {
      int relY = gy - WAVEFORM_Y_START;

      if (majorSpacing > 0 && clearX % majorSpacing == 0 && gy % 2 == 0)
        tft.drawPixel(clearX, gy, GRID_MAJOR);
      else if (minorSpacing > 0 && clearX % minorSpacing == 0 &&
               gy % 4 == 0)
        tft.drawPixel(clearX, gy, GRID_MINOR);

      if (relY % 40 == 0 && clearX % 2 == 0)
        tft.drawPixel(clearX, gy, GRID_MAJOR);
      else if (relY % 8 == 0 && clearX % 4 == 0)
        tft.drawPixel(clearX, gy, GRID_MINOR);
    }
  }

  // Draw the waveform line
  if (hasPreviousSample && pixelsCrossed > 0 &&
      pixelsCrossed < TFT_WIDTH / 2) {
    if (newPixelX >= prevPixelX) {
      tft.drawLine(prevPixelX, prevY, newPixelX, y, WAVEFORM_COLOR);
      int prevY2 = constrain(prevY + 1,
                             WAVEFORM_Y_START, WAVEFORM_Y_END);
      int y2     = constrain(y + 1,
                             WAVEFORM_Y_START, WAVEFORM_Y_END);
      tft.drawLine(prevPixelX, prevY2, newPixelX, y2, WAVEFORM_COLOR);
    } else {
      int totalDist  = (TFT_WIDTH - prevPixelX) + newPixelX;
      int distToEdge = TFT_WIDTH - 1 - prevPixelX;
      int edgeY;
      if (totalDist > 0)
        edgeY = prevY +
                (int)((long)(y - prevY) * distToEdge / totalDist);
      else
        edgeY = prevY;
      edgeY = constrain(edgeY, WAVEFORM_Y_START, WAVEFORM_Y_END);

      tft.drawLine(prevPixelX, prevY,
                   TFT_WIDTH - 1, edgeY, WAVEFORM_COLOR);
      tft.drawLine(0, edgeY,
                   newPixelX, y, WAVEFORM_COLOR);
      int prevY2 = constrain(prevY + 1,
                             WAVEFORM_Y_START, WAVEFORM_Y_END);
      int edgeY2 = constrain(edgeY + 1,
                             WAVEFORM_Y_START, WAVEFORM_Y_END);
      int y2     = constrain(y + 1,
                             WAVEFORM_Y_START, WAVEFORM_Y_END);
      tft.drawLine(prevPixelX, prevY2,
                   TFT_WIDTH - 1, edgeY2, WAVEFORM_COLOR);
      tft.drawLine(0, edgeY2,
                   newPixelX, y2, WAVEFORM_COLOR);
    }
  }

  prevSampleY       = y;
  lastWaveformY     = y;
  hasPreviousSample = true;
  scrollX           = newPixelX;

  // Update header periodically
  static int sampleCount = 0;
  if (sampleCount++ % 100 == 0) {
    tft.fillRect(0, 0, TFT_WIDTH, WAVEFORM_Y_START - 2, TFT_BLACK);
    tft.setTextSize(1);

    tft.setTextColor(TFT_CYAN);
    tft.setCursor(5, 5);
    tft.print("ECG Live");

    tft.setTextColor(TFT_YELLOW);
    tft.setCursor(80, 5);
    tft.print("HR:");
    tft.print(bpm);

    tft.setTextColor(signalQuality >= 80 ? TFT_GREEN
                   : (signalQuality >= 50 ? TFT_YELLOW : TFT_RED));
    tft.setCursor(140, 5);
    tft.print("Q:");
    tft.print(signalQuality);
    tft.print("%");

    tft.setTextColor(TFT_CYAN);
    tft.setCursor(5, 15);
    tft.print("Zoom:");
    tft.print(ZOOM_LEVELS[currentZoomLevel], 1);
    tft.print("s");

    tft.setTextColor(TFT_WHITE);
    tft.setCursor(80, 15);
    tft.print(measuredRate, 0);
    tft.print("Hz");

    tft.setTextColor(TFT_MAGENTA);
    tft.setCursor(200, 5);
    tft.print("Beats:");
    tft.print(beatCount);

    tft.setTextColor(TFT_WHITE);
    tft.setCursor(130, 15);
    tft.print("Pos:");
    if (waveformVerticalOffset >= 0) tft.print("+");
    tft.print(waveformVerticalOffset);

    tft.setTextColor(TFT_DARKGREY);
    tft.setCursor(200, 15);
    tft.print("UP/DN:Move");
  }

  // Draw position indicator on right edge
  static int lastIndicatorY = -1;
  int indicatorY = WAVEFORM_CENTER + waveformVerticalOffset;
  indicatorY = constrain(indicatorY, WAVEFORM_Y_START + 2,
                         WAVEFORM_Y_END - 2);

  if (indicatorY != lastIndicatorY) {
    if (lastIndicatorY >= WAVEFORM_Y_START &&
        lastIndicatorY <= WAVEFORM_Y_END) {
      tft.fillRect(TFT_WIDTH - 4, lastIndicatorY - 2, 4, 5, TFT_BLACK);
    }
    tft.drawPixel(TFT_WIDTH - 1, indicatorY, TFT_YELLOW);
    tft.drawPixel(TFT_WIDTH - 2, indicatorY - 1, TFT_YELLOW);
    tft.drawPixel(TFT_WIDTH - 2, indicatorY, TFT_YELLOW);
    tft.drawPixel(TFT_WIDTH - 2, indicatorY + 1, TFT_YELLOW);
    tft.drawPixel(TFT_WIDTH - 3, indicatorY - 2, TFT_YELLOW);
    tft.drawPixel(TFT_WIDTH - 3, indicatorY + 2, TFT_YELLOW);
    lastIndicatorY = indicatorY;
  }
}

void drawAICollectionScreen() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(60, 10);
  tft.print("AI ANALYSIS");

  tft.setTextSize(1);
  tft.setTextColor(TFT_YELLOW);
  tft.setCursor(10, 50);
  tft.print("Collecting ECG samples...");
  tft.setCursor(10, 70);
  tft.print("Please remain still.");

  tft.setTextColor(TFT_CYAN);
  tft.setCursor(10, 100);
  tft.print("Time remaining:");

  tft.fillRect(10, 120, 300, 20, TFT_DARKGREY);

  tft.setTextColor(TFT_WHITE);
  tft.setCursor(10, 145);
  tft.print("10 seconds");

  tft.setCursor(10, 170);
  tft.setTextColor(TFT_YELLOW);
  tft.print("BPM: ");    tft.print(bpm);
  tft.print(" | Quality: "); tft.print(signalQuality);
  tft.print("%");
}

void updateAICollectionProgress() {
  int barWidth = (aiCollectionProgress * 300) / 100;
  tft.fillRect(10, 120, 300, 20, TFT_DARKGREY);
  tft.fillRect(10, 120, barWidth, 20, TFT_GREEN);

  unsigned long elapsed   = millis() - aiCollectionStartTime;
  unsigned long remaining = (AI_COLLECTION_DURATION - elapsed) / 1000;

  tft.fillRect(10, 145, 100, 10, TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(10, 145);
  tft.print(remaining);
  tft.print(" seconds");

  tft.fillRect(10, 190, 150, 10, TFT_BLACK);
  tft.setTextColor(TFT_GREEN);
  tft.setCursor(10, 190);
  tft.print("Progress: ");
  tft.print(aiCollectionProgress);
  tft.print("%");

  tft.fillRect(10, 170, 250, 10, TFT_BLACK);
  tft.setTextColor(TFT_YELLOW);
  tft.setCursor(10, 170);
  tft.print("BPM: ");  tft.print(bpm);
  tft.print(" | Q: "); tft.print(signalQuality);
  tft.print("%");

  tft.fillRect(10, 210, 250, 10, TFT_BLACK);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(10, 210);
  tft.print("Samples: "); tft.print(aiEcgWriteIndex);
  tft.print(" | R-peaks: "); tft.print(rpeakCount);
}

void drawAIAnalysis() {
  static int  lastBPM           = -1;
  static int  lastSignalQuality = -1;
  static bool firstDraw         = true;

  if (displayNeedsFullRedraw || firstDraw) {
    tft.fillScreen(TFT_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(TFT_WHITE);
    tft.setCursor(60, 5);
    tft.print("AI ANALYSIS");

    tft.setTextSize(1);
    tft.setTextColor(TFT_YELLOW);
    tft.setCursor(10, 35);
    tft.print("Heart Rate:");
    tft.setCursor(10, 55);
    tft.print("Signal Quality:");

    tft.setTextColor(TFT_CYAN);
    tft.setCursor(10, 75);
    tft.print("Beats analyzed: ");
    tft.print(classifiedBeatCount);
    tft.print(" (");
    tft.print(rejectedBeatCount);
    tft.print(" rejected)");

    tft.setCursor(10, 95);
    tft.setTextSize(2);
    tft.setTextColor(TFT_WHITE);
    tft.print("Classification:");

    if (aiStatusMessage.length() > 0) {
      tft.setTextSize(1);
      tft.setTextColor(TFT_ORANGE);
      tft.setCursor(10, 220);
      tft.print(aiStatusMessage);
    }

    displayNeedsFullRedraw = false;
    firstDraw              = false;
    lastBPM                = -1;
    lastSignalQuality      = -1;
  }

  if (bpm != lastBPM) {
    tft.fillRect(150, 35, 150, 10, TFT_BLACK);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE);
    tft.setCursor(150, 35);
    tft.print(String(bpm) + " BPM");
    lastBPM = bpm;
  }

  if (signalQuality != lastSignalQuality) {
    tft.fillRect(150, 55, 150, 10, TFT_BLACK);
    tft.setCursor(150, 55);
    if (signalQuality >= 80) tft.setTextColor(TFT_GREEN);
    else if (signalQuality >= 50) tft.setTextColor(TFT_YELLOW);
    else tft.setTextColor(TFT_RED);
    tft.print(String(signalQuality) + "%");
    lastSignalQuality = signalQuality;
  }

  tft.fillRect(0, 115, 320, 100, TFT_BLACK);
  tft.setTextSize(1);

  for (int i = 0; i < NUM_CLASSES; i++) {
    int col = i < 2 ? 0 : 1;
    int row = i % 2;
    int x   = 15 + col * 160;
    int y   = 115 + row * 45;

    tft.fillRect(x, y, 140, 40, TFT_BLACK);

    uint16_t textColor, barColor;
    if (i == 0) {
      textColor = TFT_GREEN; barColor = TFT_GREEN;
    } else if (currentECG.classProbs[i] > 20.0f) {
      textColor = TFT_RED; barColor = TFT_RED;
    } else if (currentECG.classProbs[i] > 5.0f) {
      textColor = TFT_YELLOW; barColor = TFT_YELLOW;
    } else {
      textColor = TFT_CYAN; barColor = TFT_CYAN;
    }

    tft.setTextColor(textColor);
    tft.setCursor(x, y);
    tft.print(CLASS_ABBREV[i]);
    tft.print(":");

    tft.setTextColor(TFT_WHITE);
    tft.setCursor(x, y + 12);
    tft.print(currentECG.classProbs[i], 1);
    tft.print("%");

    int barLen = (int)(currentECG.classProbs[i] * 120.0f / 100.0f);
    barLen     = constrain(barLen, 0, 120);
    tft.fillRect(x, y + 24, 120, 6, TFT_DARKGREY);
    tft.fillRect(x, y + 24, barLen, 6, barColor);
  }

  int   dominantClass = 0;
  float maxProb       = currentECG.classProbs[0];
  for (int c = 1; c < NUM_CLASSES; c++) {
    if (currentECG.classProbs[c] > maxProb) {
      maxProb       = currentECG.classProbs[c];
      dominantClass = c;
    }
  }

  tft.fillRect(0, 210, 320, 30, TFT_BLACK);
  tft.setTextSize(1);

  if (maxProb >= CONFIDENCE_THRESHOLD * 100.0f) {
    uint16_t resultColor = (dominantClass == 0) ? TFT_GREEN : TFT_RED;
    tft.setTextColor(resultColor);
    tft.setCursor(10, 212);
    tft.print("Result: ");
    tft.setTextSize(2);
    tft.print(CLASS_NAMES[dominantClass]);
    tft.setTextSize(1);
    tft.print(" (");
    tft.print(maxProb, 1);
    tft.print("%)");
  } else {
    tft.setTextColor(TFT_ORANGE);
    tft.setCursor(10, 212);
    tft.print("Result: Inconclusive - please retry");
  }

  tft.setTextColor(TFT_DARKGREY);
  tft.setCursor(10, 228);
  tft.print("Beats: ");
  tft.print(classifiedBeatCount);
  tft.print(" analyzed | ");
  tft.print(rejectedBeatCount);
  tft.print(" rejected");
}

// ============================================================
// BUTTON HANDLING
// ============================================================

void task_Button() { handleButtons(); }

void handleButtons() {
  static bool upPressed     = false;
  static bool downPressed   = false;
  static bool selectPressed = false;
  static bool exitPressed   = false;

  static unsigned long upHoldStart    = 0;
  static unsigned long downHoldStart  = 0;
  static unsigned long lastUpRepeat   = 0;
  static unsigned long lastDownRepeat = 0;
  const  unsigned long REPEAT_DELAY   = 400;
  const  unsigned long REPEAT_RATE    = 80;

  bool upCurrent     = digitalRead(BUTTON_UP)     == LOW;
  bool downCurrent   = digitalRead(BUTTON_DOWN)   == LOW;
  bool selectCurrent = digitalRead(BUTTON_SELECT) == LOW;
  bool exitCurrent   = digitalRead(BUTTON_EXIT)   == LOW;

  unsigned long now = millis();

  // ============ UP BUTTON ============
  if (upCurrent) {
    if (!upPressed) {
      upPressed        = true;
      upHoldStart      = now;
      lastUpRepeat     = now;
      lastActivityTime = now;
      digitalWrite(STM32_WAKEUP_PIN, LOW);
      delay(50);
      digitalWrite(STM32_WAKEUP_PIN, HIGH);

      if (currentDisplayMode == DISPLAY_MAIN_MENU) {
        mainMenuSelection = (mainMenuSelection - 1 + 2) % 2;
        displayNeedsFullRedraw = true;
      }
      else if (currentDisplayMode == DISPLAY_ECG_WAVEFORM) {
        if (selectCurrent) {
          if (currentZoomLevel < NUM_ZOOM_LEVELS - 1) {
            currentZoomLevel++;
            displayNeedsFullRedraw = true;
          }
        } else {
          waveformVerticalOffset -= VERTICAL_OFFSET_STEP;
          if (waveformVerticalOffset < VERTICAL_OFFSET_MIN)
            waveformVerticalOffset = VERTICAL_OFFSET_MIN;
          Serial.printf("Vertical offset: %d\n",
                        waveformVerticalOffset);
        }
      }
    } else {
      if (currentDisplayMode == DISPLAY_ECG_WAVEFORM &&
          !selectCurrent) {
        if (now - upHoldStart > REPEAT_DELAY &&
            now - lastUpRepeat > REPEAT_RATE) {
          lastUpRepeat = now;
          waveformVerticalOffset -= VERTICAL_OFFSET_STEP;
          if (waveformVerticalOffset < VERTICAL_OFFSET_MIN)
            waveformVerticalOffset = VERTICAL_OFFSET_MIN;
        }
      }
    }
  } else {
    upPressed = false;
  }

  // ============ DOWN BUTTON ============
  if (downCurrent) {
    if (!downPressed) {
      downPressed      = true;
      downHoldStart    = now;
      lastDownRepeat   = now;
      lastActivityTime = now;
      digitalWrite(STM32_WAKEUP_PIN, LOW);
      delay(50);
      digitalWrite(STM32_WAKEUP_PIN, HIGH);

      if (currentDisplayMode == DISPLAY_MAIN_MENU) {
        mainMenuSelection = (mainMenuSelection + 1) % 2;
        displayNeedsFullRedraw = true;
      }
      else if (currentDisplayMode == DISPLAY_ECG_WAVEFORM) {
        if (selectCurrent) {
          if (currentZoomLevel > 0) {
            currentZoomLevel--;
            displayNeedsFullRedraw = true;
          }
        } else {
          waveformVerticalOffset += VERTICAL_OFFSET_STEP;
          if (waveformVerticalOffset > VERTICAL_OFFSET_MAX)
            waveformVerticalOffset = VERTICAL_OFFSET_MAX;
          Serial.printf("Vertical offset: %d\n",
                        waveformVerticalOffset);
        }
      }
    } else {
      if (currentDisplayMode == DISPLAY_ECG_WAVEFORM &&
          !selectCurrent) {
        if (now - downHoldStart > REPEAT_DELAY &&
            now - lastDownRepeat > REPEAT_RATE) {
          lastDownRepeat = now;
          waveformVerticalOffset += VERTICAL_OFFSET_STEP;
          if (waveformVerticalOffset > VERTICAL_OFFSET_MAX)
            waveformVerticalOffset = VERTICAL_OFFSET_MAX;
        }
      }
    }
  } else {
    downPressed = false;
  }

  // ============ SELECT BUTTON ============
  if (selectCurrent && !selectPressed) {
    selectPressed    = true;
    lastActivityTime = now;
    digitalWrite(STM32_WAKEUP_PIN, LOW);
    delay(50);
    digitalWrite(STM32_WAKEUP_PIN, HIGH);

    if (currentDisplayMode == DISPLAY_MAIN_MENU) {
      if (mainMenuSelection == 0) {
        currentDisplayMode     = DISPLAY_ECG_WAVEFORM;
        displayNeedsFullRedraw = true;
      } else {
        startAICollection();
      }
    }
  } else if (!selectCurrent) {
    selectPressed = false;
  }

  // ============ EXIT BUTTON ============
  if (exitCurrent && !exitPressed) {
    exitPressed      = true;
    lastActivityTime = now;
    digitalWrite(STM32_WAKEUP_PIN, LOW);
    delay(50);
    digitalWrite(STM32_WAKEUP_PIN, HIGH);

    if (currentDisplayMode != DISPLAY_MAIN_MENU) {
      currentDisplayMode      = DISPLAY_MAIN_MENU;
      aiCollectionInProgress  = false;
      aiBufferCollecting      = false;
      aiResultsReady          = false;
      displayNeedsFullRedraw  = true;
    }
  } else if (!exitCurrent) {
    exitPressed = false;
  }
}

void task_HR_Update() {}

void task_SleepCheck() {
  if (systemActive &&
      (millis() - lastActivityTime >= SLEEP_TIMEOUT_MS)) {
    systemActive = false;
    enterSleepMode();
  }
}

// ===== MAIN LOOP =====
void loop() {
  unsigned long now = millis();

  if (now - tasks[TASK_UART_READ].lastRun >=
      tasks[TASK_UART_READ].interval) {
    tasks[TASK_UART_READ].lastRun = now;
    task_UART_Read();
  }
  if (now - tasks[TASK_PROCESS_HEART_RATE].lastRun >=
      tasks[TASK_PROCESS_HEART_RATE].interval) {
    tasks[TASK_PROCESS_HEART_RATE].lastRun = now;
    task_ProcessHeartRate();
  }
  if (now - tasks[TASK_DISPLAY].lastRun >=
      tasks[TASK_DISPLAY].interval) {
    tasks[TASK_DISPLAY].lastRun = now;
    task_Display();
  }
  if (now - tasks[TASK_BUTTON].lastRun >=
      tasks[TASK_BUTTON].interval) {
    tasks[TASK_BUTTON].lastRun = now;
    task_Button();
  }
  if (now - tasks[TASK_HR_UPDATE].lastRun >=
      tasks[TASK_HR_UPDATE].interval) {
    tasks[TASK_HR_UPDATE].lastRun = now;
    task_HR_Update();
  }
  if (now - tasks[TASK_SLEEP_CHECK].lastRun >=
      tasks[TASK_SLEEP_CHECK].interval) {
    tasks[TASK_SLEEP_CHECK].lastRun = now;
    task_SleepCheck();
  }
  if (now - tasks[TASK_RATE_REPORT].lastRun >=
      tasks[TASK_RATE_REPORT].interval) {
    tasks[TASK_RATE_REPORT].lastRun = now;
    task_RateReport();
  }
  if (now - tasks[TASK_AI_COLLECTION].lastRun >=
      tasks[TASK_AI_COLLECTION].interval) {
    tasks[TASK_AI_COLLECTION].lastRun = now;
    task_AICollection();
  }

  delay(1);
}