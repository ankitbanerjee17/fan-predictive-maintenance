#include <Wire.h>

// ── MPU6050 registers ─────────────────────────────────────
#define MPU_ADDR     0x68
#define PWR_MGMT_1   0x6B
#define SMPLRT_DIV   0x19
#define CONFIG_REG   0x1A
#define ACCEL_CONFIG 0x1C
#define ACCEL_XOUT_H 0x3B

// ── Sampling config ───────────────────────────────────────
#define WINDOW_SIZE  1024
#define BAUD_RATE    460800
#define SAMPLE_HZ    1000
#define SAMPLE_US    1000

// ── Scale (G units) ───────────────────────────────────────
const float SCALE = 1.0f / 16384.0f;

// ── Buffers ───────────────────────────────────────────────
float ax_buf[WINDOW_SIZE];
float ay_buf[WINDOW_SIZE];
float az_buf[WINDOW_SIZE];

int  sample_idx   = 0;
bool window_ready = false;

// ── Software sampling timer ───────────────────────────────
unsigned long last_sample_us = 0;

// ── MPU6050 init ──────────────────────────────────────────
void mpu_write(uint8_t reg, uint8_t val)
{
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission(true);
}

void mpu_init()
{
    Wire.begin(21, 22, 400000);

    delay(100);

    mpu_write(PWR_MGMT_1,   0x00);
    mpu_write(SMPLRT_DIV,   0x00);
    mpu_write(CONFIG_REG,   0x01);
    mpu_write(ACCEL_CONFIG, 0x00);

    delay(50);
}

// ── Read accelerometer ────────────────────────────────────
void read_accel()
{
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(ACCEL_XOUT_H);
    Wire.endTransmission(false);

    Wire.requestFrom(MPU_ADDR, 6, true);

    int16_t rx = (Wire.read() << 8) | Wire.read();
    int16_t ry = (Wire.read() << 8) | Wire.read();
    int16_t rz = (Wire.read() << 8) | Wire.read();

    ax_buf[sample_idx] = rx * SCALE;
    ay_buf[sample_idx] = ry * SCALE;
    az_buf[sample_idx] = rz * SCALE;
}

// ── Setup ─────────────────────────────────────────────────
void setup()
{
    Serial.begin(BAUD_RATE);

    delay(1000);

    mpu_init();

    Serial.println("# Fan Fault Monitor v2 (G-units)");
    Serial.println("# Waiting for windows...");
}

// ── Loop ──────────────────────────────────────────────────
void loop()
{
    // Sample at 1000 Hz
    if (!window_ready)
    {
        if ((micros() - last_sample_us) >= SAMPLE_US)
        {
            last_sample_us += SAMPLE_US;

            read_accel();

            sample_idx++;

            if (sample_idx >= WINDOW_SIZE)
            {
                window_ready = true;
            }
        }

        return;
    }

    // Send window
    Serial.print("WIN ");

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Serial.print(ax_buf[i], 6);
        Serial.print(' ');
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Serial.print(ay_buf[i], 6);
        Serial.print(' ');
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Serial.print(az_buf[i], 6);

        if (i < WINDOW_SIZE - 1)
            Serial.print(' ');
    }

    Serial.println();

    // Wait for prediction
    unsigned long t0 = millis();

    while (!Serial.available() &&
           (millis() - t0 < 1000))
    {
    }

    if (Serial.available())
    {
        String result =
            Serial.readStringUntil('\n');

        result.trim();

        Serial.print("# PREDICTION: ");
        Serial.println(result);
    }
    else
    {
        Serial.println("# TIMEOUT");
    }

    // Reset acquisition
    sample_idx   = 0;
    window_ready = false;
}

