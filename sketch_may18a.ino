#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"           
#include "soc/rtc_cntl_reg.h"  

// GOOGLE NATIVE LIBRARIES
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// ==========================================
// 1. NATIVE AI CONFIGURATION
// ==========================================
#define FRAME_WIDTH 96
#define FRAME_HEIGHT 96
#define NUMBER_OF_INPUTS (FRAME_WIDTH * FRAME_HEIGHT)
#define NUMBER_OF_OUTPUTS 3 

const char* labels[NUMBER_OF_OUTPUTS] = {"rock", "paper", "scissors"};

// TensorFlow Pointers
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 100KB allocated for the PSRAM
const int kTensorArenaSize = 100 * 1024;
uint8_t* tensor_arena = nullptr;

// ==========================================
// 2. FREENOVE WROVER PINS (OV3660)
// ==========================================
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  21
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    19
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    5
#define Y2_GPIO_NUM    4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

void setup_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; 
  config.frame_size = FRAMESIZE_96X96;       
  config.jpeg_quality = 12; 
  config.fb_count = 1; 

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera error.");
    while(true); 
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  setCpuFrequencyMhz(160);
  
  Serial.begin(115200);
  delay(3000);

  Serial.println("\n==================================");
  Serial.println("Step 1: Initializing native hardware...");
  setup_camera();
  Serial.println("✅ Camera OK");

  // Allocating memory in PSRAM
  tensor_arena = (uint8_t*) heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tensor_arena == nullptr) {
    Serial.println("❌ Error: Could not allocate memory in PSRAM.");
    while(true);
  }
  Serial.println("✅ PSRAM Reserved for AI.");

  // Initializing the Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  tflite_model = tflite::GetModel(g_model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Error: Incompatible model version.");
    while(true);
  }

  // Passing error_reporter as the 5th argument
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Error: AllocateTensors failed.");
    while(true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ Model loaded. Let the game begin!");
  Serial.println("==================================\n");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  if (fb->len != NUMBER_OF_INPUTS) {
    esp_camera_fb_return(fb);
    return;
  }
  
  for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
    input->data.int8[i] = (int8_t)(fb->buf[i] - 128); 
  }
  
  esp_camera_fb_return(fb);
  
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Error during inference.");
    return;
  }
  
  // Calculating real probabilities from the quantized INT8 output
  float scores[NUMBER_OF_OUTPUTS];
  int predicted_class = 0;
  float max_score = -1.0;

  for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
    scores[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    if (scores[i] > max_score) {
      max_score = scores[i];
      predicted_class = i;
    }
  }
  
  Serial.printf("[ R: %.2f | P: %.2f | S: %.2f ] ---> ", scores[0], scores[1], scores[2]);
  
  // Display high-confidence predictions (greater than 60%)
  if (max_score > 0.40) {
    if (predicted_class == 0) Serial.printf("✊ ROCK! (%.0f%%)\n", max_score * 100);
    else if (predicted_class == 1) Serial.printf("✋ PAPER! (%.0f%%)\n", max_score * 100);
    else if (predicted_class == 2) Serial.printf("✌️ SCISSORS! (%.0f%%)\n", max_score * 100);
  } else {
    Serial.println("❓ Thinking...");
  }
  
  delay(800);
}