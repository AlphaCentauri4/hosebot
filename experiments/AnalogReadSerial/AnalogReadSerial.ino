//* The ATmega2560 has a 10-bit ADC, and each analogRead()
//*  takes roughly 100 µs under the standard Arduino configuration. 
//*  Two reads therefore consume around 200 µs before serial transmission 
//*  and program overhead, making 1 kHz a reasonable starting target. 
//*  The exact sustainable rate depends heavily on the number of digits transmitted


const unsigned long SAMPLE_RATE_HZ = 1000;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

unsigned long experimentStartUs;
unsigned long nextSampleUs;

void setup() {
  Serial.begin(250000);

  experimentStartUs = micros();
  nextSampleUs = experimentStartUs;
}

void loop() {
  unsigned long now = micros();

  if ((long)(now - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_PERIOD_US;

    int sensorValue0 = analogRead(A0);
    int sensorValue1 = analogRead(A1);
    int sensorValue2 = analogRead(A2);
    int sensorValue3 = analogRead(A5);
    int sensorValue4 = analogRead(A13);

    // Time elapsed since experiment start, in microseconds
    unsigned long elapsedUs = now - experimentStartUs;

    Serial.print(elapsedUs);
    Serial.write('\t');
    Serial.print(sensorValue0);
    Serial.write('\t');
    Serial.print(sensorValue1);
    Serial.write('\t');
    Serial.print(sensorValue2);
    Serial.write('\t');
    Serial.print(sensorValue3);
    Serial.write('\t');
    Serial.println(sensorValue4);
  }
}