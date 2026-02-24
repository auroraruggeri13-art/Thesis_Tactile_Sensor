/*
 * DPS310 Multi-Sensor Datalogger — MKR ZERO Version (Safe, Non-Blocking)
 * - 6x DPS310 over SPI
 * - Reads BOTH pressure and temperature
 * - Proper manual CS control (critical!)
 * - Robust for ROS serial streaming
 */

#include <Adafruit_DPS310.h>
#include <SPI.h>

//========================= CONFIG =========================
const int CS_PINS[] = {1, 2, 3, 4, 5, 0};
const int NUM_SENSORS = sizeof(CS_PINS) / sizeof(CS_PINS[0]);
const unsigned long READING_INTERVAL_MS = 10; // ~100 Hz
const uint32_t SERIAL_BAUD = 115200;
//=========================================================

Adafruit_DPS310 dps[NUM_SENSORS];
unsigned long last_read_ms = 0;

void setup() {

  Serial.begin(SERIAL_BAUD);
  while (!Serial) { ; }
  delay(200);

  Serial.println("=== MKRZERO DPS310 LOGGER ===");
  Serial.println("Booting...");

  // -------------------------------
  // CS pin setup
  // -------------------------------
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);   // disable all sensors
  }

  // -------------------------------
  // OPTIONAL HANDSHAKE
  // -------------------------------
  Serial.println("Waiting for host handshake (2s timeout)...");
  unsigned long t0 = millis();
  bool handshake = false;
  while (millis() - t0 < 2000) {
    if (Serial.available()) {
      Serial.read();
      handshake = true;
      break;
    }
    delay(5);
  }
  if (handshake) Serial.println("Handshake received.");
  else Serial.println("No handshake — continuing.");

  // -------------------------------
  // SENSOR INITIALIZATION
  // -------------------------------
  Serial.println("Initializing sensors...");

  for (int i = 0; i < NUM_SENSORS; i++) {

    Serial.print("DPS310 #");
    Serial.print(i+1);
    Serial.print(" CS=");
    Serial.print(CS_PINS[i]);
    Serial.print(" ... ");

    bool ok = false;
    unsigned long t_start = millis();

    while (!ok && millis() - t_start < 2000) {
      ok = dps[i].begin_SPI(CS_PINS[i]);
      if (!ok) delay(50);
    }

    if (!ok) {
      Serial.println("FAILED!");
      continue;
    }

    Serial.println("OK");

    dps[i].configurePressure(DPS310_64HZ, DPS310_2SAMPLES);
    dps[i].configureTemperature(DPS310_4HZ, DPS310_2SAMPLES);
  }

  Serial.println("Init complete.");

  // HEADER
  Serial.print("Time(ms)");
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(",B");
    Serial.print(i+1);
    Serial.print("_P,B");
    Serial.print(i+1);
    Serial.print("_T");
  }
  Serial.println();
}

void loop() {

  unsigned long now = millis();
  if (now - last_read_ms < READING_INTERVAL_MS) return;
  last_read_ms = now;

  Serial.print(now);

  for (int i = 0; i < NUM_SENSORS; i++) {

    sensors_event_t temp_event, pressure_event;

    // ----------- CRITICAL FIX -----------
    // Only ONE DPS310 active at a time
    digitalWrite(CS_PINS[i], LOW); // enable sensor
    bool ok = dps[i].getEvents(&temp_event, &pressure_event);
    digitalWrite(CS_PINS[i], HIGH); // disable sensor
    // ------------------------------------

    Serial.print(',');

    if (ok) {
      Serial.print(pressure_event.pressure, 2);
      Serial.print(',');
      Serial.print(temp_event.temperature, 2);
    } else {
      Serial.print("NaN,NaN");
    }
  }

  Serial.println();
}
