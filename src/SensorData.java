import java.time.Instant;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable carrier objects for raw sensor data from each modality on the rover.
 * All sensor frames are time-stamped using hardware PPS signals (jitter < 0.5 ms).
 */
public final class SensorData {
 
    private SensorData() {}  // utility namespace — not instantiated directly
 
    // =========================================================================
    //  1. RGB-D Frame — ZED 2 Stereo Camera
    // =========================================================================
 
    /**
     * One stereo pair from the ZED 2 at up to 100 fps (1280 × 720 per eye).
     * The depth map is produced by the ZED SDK (SGM + neural depth refinement).
     */
    public record RgbdFrame(
            Instant  timestamp,
            int      frameId,
            byte[]   leftRgb,       // raw BGR bytes, width × height × 3
            byte[]   rightRgb,
            float[]  depthMetres,   // width × height, NaN where invalid
            int      width,
            int      height,
            float    fxLeft,        // focal length (pixels)
            float    fyLeft,
            float    cxLeft,        // principal point
            float    cyLeft
    ) {
        public RgbdFrame {
            Objects.requireNonNull(timestamp, "timestamp");
            Objects.requireNonNull(leftRgb,   "leftRgb");
            Objects.requireNonNull(depthMetres,"depthMetres");
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid frame dimensions");
        }
 
        /** Returns the depth value (metres) at pixel (u, v), or NaN if invalid. */
        public float depthAt(int u, int v) {
            if (u < 0 || u >= width || v < 0 || v >= height) return Float.NaN;
            return depthMetres[v * width + u];
        }
 
        /** Back-projects a pixel to a 3D point in camera frame (Z forward). */
        public double[] backProject(int u, int v) {
            float z = depthAt(u, v);
            if (Float.isNaN(z)) return null;
            return new double[]{
                (u - cxLeft) * z / fxLeft,
                (v - cyLeft) * z / fyLeft,
                z
            };
        }
    }
 
    // =========================================================================
    //  2. LiDAR Point Cloud — Livox Mid-360
    // =========================================================================
 
    /**
     * One sweep from the Livox Mid-360 (non-repetitive scan pattern, 360° horizontal).
     * Points are in the LiDAR's own coordinate frame; extrinsic calibration is
     * applied downstream in the preprocessing node.
     */
    public static final class PointCloud {
 
        private final Instant  timestamp;
        private final int      sweepId;
        private final float[]  xs;   // parallel arrays, length == numPoints
        private final float[]  ys;
        private final float[]  zs;
        private final float[]  intensities;
        private final int      numPoints;
 
        public PointCloud(Instant timestamp, int sweepId,
                          float[] xs, float[] ys, float[] zs, float[] intensities) {
            this.timestamp  = Objects.requireNonNull(timestamp, "timestamp");
            this.sweepId    = sweepId;
            this.xs         = Arrays.copyOf(xs, xs.length);
            this.ys         = Arrays.copyOf(ys, ys.length);
            this.zs         = Arrays.copyOf(zs, zs.length);
            this.intensities = Arrays.copyOf(intensities, intensities.length);
            this.numPoints  = xs.length;
            if (ys.length != numPoints || zs.length != numPoints)
                throw new IllegalArgumentException("Mismatched point cloud array lengths");
        }
 
        public Instant getTimestamp()   { return timestamp; }
        public int     getSweepId()     { return sweepId; }
        public int     getNumPoints()   { return numPoints; }
 
        public float x(int i)         { return xs[i]; }
        public float y(int i)         { return ys[i]; }
        public float z(int i)         { return zs[i]; }
        public float intensity(int i) { return intensities[i]; }
 
        /** Euclidean range from sensor origin to point i. */
        public float range(int i) {
            return (float) Math.sqrt(xs[i]*xs[i] + ys[i]*ys[i] + zs[i]*zs[i]);
        }
 
        /**
         * Returns a filtered copy keeping only points within [minRange, maxRange] metres
         * and above {@code minHeightM} in the Z axis.
         */
        public PointCloud filtered(float minRange, float maxRange, float minHeightM) {
            int count = 0;
            boolean[] keep = new boolean[numPoints];
            for (int i = 0; i < numPoints; i++) {
                float r = range(i);
                keep[i] = (r >= minRange && r <= maxRange && zs[i] >= minHeightM);
                if (keep[i]) count++;
            }
            float[] fx = new float[count], fy = new float[count],
                    fz = new float[count], fi = new float[count];
            int j = 0;
            for (int i = 0; i < numPoints; i++) {
                if (keep[i]) { fx[j]=xs[i]; fy[j]=ys[i]; fz[j]=zs[i]; fi[j]=intensities[i]; j++; }
            }
            return new PointCloud(timestamp, sweepId, fx, fy, fz, fi);
        }
    }
 
    // =========================================================================
    //  3. IMU + GNSS Frame — VectorNav VN-300
    // =========================================================================
 
    /**
     * One IMU reading at up to 800 Hz.  Orientation is output by the VN-300's
     * internal 18-state Kalman filter (fused gyro + accel + magnetometer + GNSS).
     */
    public record ImuFrame(
            Instant  timestamp,
            long     sequenceNumber,
 
            // Linear acceleration (m/s²), body frame
            double   accelX, double accelY, double accelZ,
 
            // Angular velocity (rad/s), body frame
            double   gyroX,  double gyroY,  double gyroZ,
 
            // Orientation as quaternion (w, x, y, z) — NED to body
            double   quatW,  double quatX,  double quatY, double quatZ,
 
            // GNSS (may be NaN if fix lost)
            double   latitudeDeg, double longitudeDeg, double altitudeM,
            float    posAccuracyM
    ) {
        /** Roll in radians derived from quaternion. */
        public double rollRad() {
            return Math.atan2(2*(quatW*quatX + quatY*quatZ),
                              1 - 2*(quatX*quatX + quatY*quatY));
        }
        /** Pitch in radians derived from quaternion. */
        public double pitchRad() {
            return Math.asin(Math.clamp(2*(quatW*quatY - quatZ*quatX), -1.0, 1.0));
        }
        /** Yaw in radians derived from quaternion. */
        public double yawRad() {
            return Math.atan2(2*(quatW*quatZ + quatX*quatY),
                              1 - 2*(quatY*quatY + quatZ*quatZ));
        }
        public boolean hasGnssFix() { return !Double.isNaN(latitudeDeg); }
    }
 
    // =========================================================================
    //  4. Thermal IR Frame — FLIR Boson+ LWIR
    // =========================================================================
 
    /**
     * One thermal frame from the FLIR Boson+ (640 × 512, 7.5–14 μm LWIR).
     * Used as fallback for low-light and heavy-dust scenarios.
     * Radiometric data is raw 14-bit sensor counts; apply calibration offset/gain
     * to convert to temperature in Kelvin: T_K = (raw * gain) + offset.
     */
    public record ThermalFrame(
            Instant  timestamp,
            int      frameId,
            short[]  rawCounts,    // 640 × 512, 14-bit
            int      width,
            int      height,
            float    gainKPerCount,
            float    offsetK
    ) {
        public ThermalFrame {
            Objects.requireNonNull(timestamp, "timestamp");
            Objects.requireNonNull(rawCounts, "rawCounts");
        }
 
        /** Returns temperature in Kelvin at pixel (u, v). */
        public float tempKelvin(int u, int v) {
            return (rawCounts[v * width + u] & 0xFFFF) * gainKPerCount + offsetK;
        }
 
        /** Returns temperature in Celsius at pixel (u, v). */
        public float tempCelsius(int u, int v) {
            return tempKelvin(u, v) - 273.15f;
        }
    }
 
    // =========================================================================
    //  5. Synchronized multi-modal bundle (time-aligned ≤ 0.5 ms jitter)
    // =========================================================================
 
    /**
     * A hardware-synchronized bundle of all four sensor modalities at one
     * pipeline tick (30 Hz).  At least {@code rgbdFrame} and {@code imuFrame}
     * must be present; the others are optional (graceful degradation).
     */
    public static final class SyncBundle {
 
        private final Instant       bundleTime;
        private final RgbdFrame     rgbdFrame;
        private final PointCloud    pointCloud;     // nullable
        private final ImuFrame      imuFrame;
        private final ThermalFrame  thermalFrame;   // nullable
 
        private SyncBundle(Builder b) {
            this.bundleTime   = Objects.requireNonNull(b.bundleTime,  "bundleTime");
            this.rgbdFrame    = Objects.requireNonNull(b.rgbdFrame,   "rgbdFrame");
            this.imuFrame     = Objects.requireNonNull(b.imuFrame,    "imuFrame");
            this.pointCloud   = b.pointCloud;
            this.thermalFrame = b.thermalFrame;
        }
 
        public Instant      getBundleTime()   { return bundleTime; }
        public RgbdFrame    getRgbdFrame()    { return rgbdFrame; }
        public ImuFrame     getImuFrame()     { return imuFrame; }
        public PointCloud   getPointCloud()   { return pointCloud; }      // may be null
        public ThermalFrame getThermalFrame() { return thermalFrame; }    // may be null
 
        public boolean hasLidar()   { return pointCloud   != null; }
        public boolean hasThermal() { return thermalFrame != null; }
 
        /** Number of active modalities in this bundle. */
        public int modalityCount() {
            return 2 + (hasLidar() ? 1 : 0) + (hasThermal() ? 1 : 0);
        }
 
        public static Builder builder() { return new Builder(); }
 
        public static final class Builder {
            private Instant       bundleTime;
            private RgbdFrame     rgbdFrame;
            private PointCloud    pointCloud;
            private ImuFrame      imuFrame;
            private ThermalFrame  thermalFrame;
 
            public Builder bundleTime(Instant t)       { this.bundleTime   = t; return this; }
            public Builder rgbdFrame(RgbdFrame f)      { this.rgbdFrame    = f; return this; }
            public Builder pointCloud(PointCloud c)    { this.pointCloud   = c; return this; }
            public Builder imuFrame(ImuFrame f)        { this.imuFrame     = f; return this; }
            public Builder thermalFrame(ThermalFrame f){ this.thermalFrame = f; return this; }
            public SyncBundle build()                  { return new SyncBundle(this); }
        }
    }
}

