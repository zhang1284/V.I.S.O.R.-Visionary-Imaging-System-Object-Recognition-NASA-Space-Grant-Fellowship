import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Objects; 

public final class DetectionResult {

    private final String           detectionId; 
    private final TerrainClass     terrainClass;
    private final BoundingBox2D    box2d;
    private final BoundingBox3D    box3d; 
    private final float            confidence;
    private final float            fusionWeight; // CMAF gate weight [0, 1]
    private final DetectionSource  source; 
    private final Instant          timestamp;

    private DetectionResult(Builder b) {
        this.detectionId = Objects.requireNonNull(b.detectionId, "detectionId");
        this.terrainClass = Objects.requireNonNull (b.terrainClass, "terrainClass");
        this.box2d = Objects.requireNonNull (b.box2d, "box2d");
        this.box3d = b.box3d; 
        this.confidence = b.confidence; 
        this.fusionWeight = b.fusionWeight; 
        this.source = Objects.requireNonNull(b.source, "source");
        this.timestamp = b.timestamp != null ? b.timestamp : Instant.now();

        
    }

    
}
