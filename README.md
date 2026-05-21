# Rock-Paper-Scissors TinyML Classification for ESP32

## Description
This project demonstrates the end-to-end pipeline for designing, optimizing, and deploying a Deep Learning model on a highly resource-constrained microcontroller (Edge AI / TinyML). 

The goal is to classify hand gestures—**Rock, Paper, and Scissors**—in real-time using an **ESP32-WROVER** development board equipped with an **OV3660 camera module**. The model is built with a compact Convolutional Neural Network (CNN) in TensorFlow/Keras, optimized with Full INT8 Quantization to **~35 KB**, and executed directly on bare-metal hardware using **Google's Native LiteRT (TensorFlow Lite Micro) API**.

---

## Step-by-Step Process

### 1. Dataset Selection & Inspection
The workflow begins with the ingestion of the Rock-Paper-Scissors dataset. The images consist of distinct hand gestures representing the three gameplay states. Exploratory data analysis (EDA) is performed to inspect dimensions and color profiles and to confirm that the classification target contains no massive statistical skewness.

<img width="1500" height="740" alt="Screenshot 2026-05-21 at 2 45 09 p m" src="https://github.com/user-attachments/assets/bc6912b5-9d37-414f-a841-b2e4af3729f5" />

### 2. Preprocessing & Stratification
To make the data viable for an embedded system, heavy optimization is applied before training:
* **Downsampling**: Images are scaled down to a strict $96 \times 96$ resolution.
* **Channel Reduction**: Images are converted to monochrome (Grayscale) to eliminate the storage and processing overhead of 3-channel RGB matrix calculations.
* **Data Splitting**: A stratified splitting algorithm divides the dataset into an 80/20 distribution for training and validation, ensuring each subset retains an identical ratio of classes to prevent algorithmic bias.

<img width="850" height="528" alt="slplit" src="https://github.com/user-attachments/assets/37888834-2918-4bda-9808-ae5f6001d069" />

### 3. Neural Network Architecture Design
A custom, hyper-compact Convolutional Neural Network (CNN) is designed. It sequences alternating 2D Convolutional layers (to extract structural edges and visual boundaries) and Max Pooling layers (to reduce spatial dimensions). Regularization techniques like Dropout are incorporated to mitigate overfitting, followed by a Dense layer that emits log-probabilities for the three target classes.

<img width="696" height="421" alt="Screenshot 2026-05-21 at 2 45 52 p m" src="https://github.com/user-attachments/assets/bdd7cd7f-b4cf-413f-9ac2-79bab0eafc89" />

### 4. Training & Early Stopping
The network is compiled using a specialized loss function for multi-class optimization and trained via the Adam optimizer. An automated Early Stopping callback monitors validation performance and immediately terminates training, preserving the absolute best weights when accuracy plateaus, saving compute cycles.

<img width="989" height="528" alt="Screenshot 2026-05-21 at 2 47 15 p m" src="https://github.com/user-attachments/assets/7ccef1f8-0001-4626-99ad-053c8f6f80c4" />

<img width="513" height="470" alt="conf" src="https://github.com/user-attachments/assets/9db63287-2958-4c72-913f-0859958ddbdc" />


### 5. Post-Training Full INT8 Quantization
The trained Keras model is pushed through the TensorFlow Lite Converter. To run efficiently on a microcontroller lacking native floating-point units (FPU), **Full Integer Quantization (INT8)** is applied. A representative data sample calibrates the numerical dynamic ranges of internal activations. This scales down weights and biases from standard 32-bit floating-points (`float32`) to compact 8-bit integers (`int8`), resulting in a **35 KB** production binary.

### 6. Firmware Implementation & Hardware Compilation
The optimized `.tflite` model is converted to a C++ hex-encoded byte array using the cross-platform formatting utility (`xxd`). The native array is integrated into an Arduino/C++ project, map-targeting internal hardware peripherals such as camera image buffers, system clocks, and the external pseudo-static RAM (PSRAM) module.

<img width="487" height="344" alt="Screenshot 2026-05-21 at 2 42 49 p m" src="https://github.com/user-attachments/assets/48a894d0-47e9-400d-bf91-eaae5fe60454" />

---

## Technical Bottlenecks Encountered

During the deployment phase, multiple critical low-level software, architecture, and hardware anomalies were identified and documented:

1.  **Library Abstract Instability & Compiler Rejection**: Initial boilerplate code used third-party wrapper wrappers (`EloquentTinyML`). This introduced directory-routing failures (`eloquent_tinyml/tf.h` missing) due to structural changes during version updates and strict macOS include-pathing guidelines.
2.  **The Strapping Pin Initialization Lock (Hardware)**: Connecting peripheral sensor wires directly to **GPIO Pin 12** caused critical physical upload failures (`Packet content transfer stopped` / `exit status 2`). GPIO 12 on the ESP32 acts as an internal boot configuration link ("Strapping Pin"). Forcing electrical current into this pin at boot drops internal flash memory voltage from 3.3V to 1.8V, completely turning off the chip's internal memory during code flashes.
3.  **Early Microcontroller Panics (Null Pointers)**: Passing early versions of the firmware triggered complete processor breakdowns (`LoadProhibited` Exception / `EXCVADDR: 0x00000010`). This happened because TensorFlow's execution engine requires its working runtime memory blocks ("Tensor Arena") to be mathematically aligned to 16-byte boundaries. Creating this arena using arbitrary C++ pointers threw memory chunks into randomized, unaligned fragments.
4.  **Internal SRAM Buffer Overflow**: Increasing the raw memory space inside standard wrapper environments caused the linker to throw fatal errors during compilation (`DRAM segment data does not fit / region dram0_0_seg overflowed`). The hardware's fast internal RAM (SRAM) was completely choked because it was simultaneously holding the camera's image frame buffer and the AI's complex matrix-calculation arena.
5.  **Modern Compiler Compliance Conflicts**: Compiling the official, older TensorFlow Lite source code on a modern desktop machine (running GCC 14 updates) resulted in direct engine syntax errors regarding a constant tracking structure (`assignment of read-only member flatbuffers::span::count_`) and memory allocation protection layers (`operator delete is private within this context`).

---

## Engineering Fixes Applied

To counter the bottlenecks and push the system into a stable production state, the following engineering patches were engineered:

* **Native Google API Transition**: The wrapper layers were stripped entirely. The firmware was rewritten to utilize the bare-metal, native **Google LiteRT (TensorFlow Lite Micro)** framework, ensuring absolute control over layer allocations and memory pointers.
* **Boot Boundary Pin Relocation**: All temporary experimental cables connected to the ESP32 strapping pins (specifically GPIO 12) were disconnected. Hardware wiring standard operating procedures were updated to limit sensor integrations strictly to unconstrained safe general-purpose pins (GPIO 13, 14, 32, 33).
* **Forced High-Capacity PSRAM Mapping**: To prevent crashing the fast internal SRAM, a memory allocation directive (`heap_caps_aligned_alloc`) was written. This bypassed the native Arduino memory allocator, forcing the system to carve a 100 KB Tensor Arena directly into the ESP32-WROVER's external 4MB **PSRAM (SPI RAM)** using explicit 16-byte memory alignment parameters (`MALLOC_CAP_SPIRAM`).
* **TensorFlow Lite Header Micro-Patching**: The internal source code of the Google framework was patched by hand to compile with modern development tools:
    * In `stl_emulation.h`, the strict `const` restriction on the memory variable `count_` was removed to allow runtime matrix modifications.
    * In `compatibility.h`, the internal virtual object eraser macro (`TF_LITE_REMOVE_VIRTUAL_DELETE`) was refactored from `private:` to `public:`. This granted the modern compiler standard permissions to access cleanup routines during structural memory builds, resolving the private declaration compiler error.

---

## Conclusions

* **TinyML Feasibility**: Edge deployment of image-processing Convolutional Neural Networks on budget microcontrollers is highly viable. Full INT8 quantization effectively reduces the memory footprint by roughly 400% without a significant sacrifice in the layers' mathematical precision.
* **Hardware Overrides Library Luxury**: Standard "easy-to-use" libraries fail when confronted with complex matrix operations or computer vision tasks. Developing stable embedded AI systems requires direct interaction with hardware features (like explicit PSRAM allocation) and standard native APIs.
* **The Environment Alignment Trap**: Machine learning models do not operate in a vacuum. A model trained on computer-generated imagery (CGI) with uniform solid backgrounds will experience sensory blindness when deployed into real-world settings (variable illumination, noisy backdrops, grayscale conversion). Real-world accuracy requires training datasets that mirror the target sensor's exact physical environment.

---

## References & Documentation Links
* **Official Google Edge Framework**: [Get Started with LiteRT on Microcontrollers](https://ai.google.dev/edge/litert/microcontrollers/get_started)
* **Dataset Resource**: [Dataset - Rock, Paper, Scissors]([https://www.tensorflow.org/datasets/catalog/rock_paper_scissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors?select=paper))
* **Hardware Architecture Guide**: [Espressif Systems ESP32-WROVER Technical Reference Manual](https://www.espressif.com/documentation/esp32_technical_reference_manual_en.pdf)
