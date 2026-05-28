import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Objects; 

public final class DetectionResult {

    private final String detectionId;
    private final TerrainClass terrainClass;
    private final BoundingBox2D box2d;
    private final BoundingBox2D.BoundingBox3D box3d;
    private final float confidence;
    private final float fusionWeight; // CMAF gate weight [0, 1]
    private final DetectionSource source;
    private final Instant timestamp;

    private DetectionResult(Builder b) {
        this.detectionId = Objects.requireNonNull(b.detectionId, "detectionId");
        this.terrainClass = Objects.requireNonNull(b.terrainClass, "terrainClass");
        this.box2d = Objects.requireNonNull(b.box2d, "box2d");
        this.box3d = b.box3d;
        this.confidence = b.confidence;
        this.fusionWeight = b.fusionWeight;
        this.source = Objects.requireNonNull(b.source, "source");
        this.timestamp = b.timestamp != null ? b.timestamp : Instant.now();
    }
    //--Accessor--------------------------------------------------------------

    public String getDetectionId() { return detectionId; }
    public TerrainClass getTerrainClass() { return terrainClass; }
    public BoundingBox2D getBox2d() { return box2d; }
    public BoundingBox2D.BoundingBox3D getBox3d() { return box3d; }
    public float getConfidence()     { return confidence; }
    public float getFusionWeight()   { return fusionWeight; }
    public DetectionSource getSource()  { return source; }
    public Instant getTimestamp() { return timestamp; }

    public boolean has3dBox() { return box3d != null; }
    public boolean isHazardous() { return terrainClass.isHazardous() && confidence >= 0.5f; }

    @Override
    public String toString() {
        return String.format("DetectionResult{id=%s class=%s conf=%.2f src=%s}",
             detectionId, terrainClass, confidence, source);
    }

    //--Builder--------------------------------------------------------------
    public static Builder builder() { return new Builder(); }
    public static final class Builder {
        private String detectionId;
        private TerrainClass terrainClass;
        private BoundingBox2D box2d;
        private BoundingBox2D.BoundingBox3D box3d;
        private float confidence;
        private float fusionWeight = 1.0f;
        private DetectionSource source = DetectionSource.FUSED;
        private Instant timestamp;

        public Builder detectionId(String id) {
            this.detectionId = id;
            return this;
        }

        public Builder terrainClass(TerrainClass tc) {
            this.terrainClass = tc;
            return this;
        }

        public Builder box2d(BoundingBox2D b) {
            this.box2d = b;
            return this;
        }

        public Builder box3d(BoundingBox2D.BoundingBox3D b) {
            this.box3d = b;
            return this;
        }

        public Builder confidence(float c) {
            this.confidence = c;
            return this;
        }

        public Builder fusionWeight(float w) {
            this.fusionWeight = w;
            return this;
        }

        public Builder source(DetectionSource s) {
            this.source = s;
            return this;
        }

        public Builder timestamp(Instant t) {
            this.timestamp = t;
            return this;
        }

        public DetectionResult build() {
            return new DetectionResult(this);
        }
    }

    //--Nested types --------------------------------------------------------------------

    /**
     * Axis aligned 2D bounding box in pixel space.
     */
    public record BoundingBox2D(int x, int y, int width, int height) {
        public BoundingBox2D {
            if (width <= 0) throw new IllegalArgumentException("width must be > 0");
            if (height <= 0) throw new IllegalArgumentException("height must be > 0");
        }

        public int cx() {
            return x + width / 2;
        }

        public int cy() {
            return y + height / 2;
        }

        public int area() {
            return width * height / 2;
        }

        public float ar() {
            return (float) width / height;
        }

        /**
         * Oriented 3D bounding box in the rover's base_link coordinate frame.
         * Origin is the geometric centroid of the box; yaw is rotation about Z-axis.
         */
        public record BoundingBox3D(double cx, double cy, double cz, double dimX, double dimY, double dimZ, double yawRad) {
            public double volume() {
                return dimX * dimY * dimZ;
            }

            public double groundArea() {
                return dimX * dimY;
            }

            public double distanceToOrigin() {
                return Math.sqrt(cx * cx + cy * cy + cz * cz);
            }
        }

        //--Batch helper ---------------------------------------------------------------------------------

        /**
         * Filters a list to only hazardous, high confidence detections.
         */
        public static List<DetectionResult> filterHazards(List<DetectionResult> results, float minConf) {
            return results.stream()
                .filter(r -> r.terrainClass.isHazardous() && r.confidence >= minConf)
                .sorted((a, b) -> Float.compare(b.confidence, a.confidence))
                .toList();
        }

        /**
         * Returns the closest detection by 3D centroid distance.
         */
        public static DetectionResult closest(List<DetectionResult> results) {
            return results.stream()
                .filter(DetectionResult::has3dBox)
                .min((a, b) -> Double.compare(a.box3d.distanceToOrigin(), b.box3d.distanceToOrigin()))
                .orElseThrow(() -> new IllegalArgumentException("No 3D detections available"));
        }
    }

    public enum DetectionSource {
        FUSED
    }
}
