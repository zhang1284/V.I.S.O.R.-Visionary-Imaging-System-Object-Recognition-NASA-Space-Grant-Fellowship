import java.nio.file.Path;

/**
 * Immutable configuration for the full Rover-CV pipeline. 
 * Build with the nested {@link Builder}; load from YAML via
 * {@code CVPipelineConfig.fromYAML(Path)} in a real deployment. 
 */
public final class CVPipelineConfig {
    //--Camera-----------------------------------------------------
    public final int     cameraWidth;
    public final int     cameraHeight;
    public final int     cameraFps;
    public final float   stereoBaseLine;  // in meters
    public final Path    intrinsicsCalibPath; 

    //--Lidar------------------------------------------------------
    public final float     lidarMinRange;   // in meters
    public final float     lidarMaxRange; 
    public final float     lidarGroundRemovalHeight; // meters below sensor origin
    public final int       lidarMaxPointsPerSweep; 

    // Constructor to initialize all final variables
    public CVPipelineConfig(int cameraWidth, int cameraHeight, int cameraFps, 
                            float stereoBaseLine, Path intrinsicsCalibPath, 
                            float lidarMinRange, float lidarMaxRange, 
                            float lidarGroundRemovalHeight, int lidarMaxPointsPerSweep) {
        this.cameraWidth = cameraWidth;
        this.cameraHeight = cameraHeight;
        this.cameraFps = cameraFps;
        this.stereoBaseLine = stereoBaseLine;
        this.intrinsicsCalibPath = intrinsicsCalibPath;
        this.lidarMinRange = lidarMinRange;
        this.lidarMaxRange = lidarMaxRange;
        this.lidarGroundRemovalHeight = lidarGroundRemovalHeight;
        this.lidarMaxPointsPerSweep = lidarMaxPointsPerSweep;
    }
} 
