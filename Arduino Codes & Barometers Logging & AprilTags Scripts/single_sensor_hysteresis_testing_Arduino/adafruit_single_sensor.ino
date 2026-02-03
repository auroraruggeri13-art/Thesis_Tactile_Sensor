#include <Wire.h>
#include <Adafruit_MPL115A2.h>

Adafruit_MPL115A2 mpl;
float tempC, pressure_kPa;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);  // wait for USB serial
  }

  // CSV header: pressure, temperature
  Serial.println("b1,t1");

  // Start I2C
  Wire.begin();

  // Initialize MPL115A2
  if (!mpl.begin()) {
    Serial.println("# ERROR: MPL115A2 not found, check wiring!");
    while (1) {
      delay(100);
    }
  }
}

void loop() {
  // Read pressure (kPa) and temperature (°C)
  mpl.getPT(&pressure_kPa, &tempC);

  // Convert kPa -> hPa to match DPS310 style (1 kPa = 10 hPa)
  float pressure_hPa = pressure_kPa * 10.0f;

  // CSV line: b1,t1
  Serial.print(pressure_hPa, 6);  // pressure in hPa
  Serial.print(',');
  Serial.println(tempC, 6);       // temperature in °C

  delay(10);  // ~100 Hz
}
