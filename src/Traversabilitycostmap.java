import edu.nasa.rovercv.model.DetectionResult;
import edu.nasa.rovercv.model.TerrainClass;
 
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
 
/**
 * Occupancy-based traversability costmap in the rover's local ground plane.
 *
 * <p>The map is a 2-D grid centered on the rover. Each cell stores a cost
 * value in [0, 255] where 0 = free / fully traversable and 255 = lethal
 * (must not enter). Costs are inflated around hazards by a Gaussian kernel
 * to provide safety margins.
 *
 * <p>Resolution: 0.10 m / cell (configurable). Default extent: ±10 m
 * ahead, ±5 m laterally → 200 × 100 cells.
 */
public final class TraversabilityCostmap {
 
    public static final int COST_FREE   = 0;
    public static final int COST_LETHAL = 255;
    public static final int COST_WARN   = 128;
 
    private final int     cols;          // x axis
    private final int     rows;          // y axis
    private final float   resolutionM;   // metres per cell
    private final float   originX;       // world X of cell (0,0)
    private final float   originY;       // world Y of cell (0,0)
    private final byte[]  costs;         // row-major, unsigned [0,255]
    private final Instant updatedAt;
 
    private TraversabilityCostmap(Builder b) {
        this.cols        = b.cols;
        this.rows        = b.rows;
        this.resolutionM = b.resolutionM;
        this.originX     = b.originX;
        this.originY     = b.originY;
        this.costs       = Arrays.copyOf(b.costs, b.costs.length);
        this.updatedAt   = b.updatedAt != null ? b.updatedAt : Instant.now();
    }
 
    // -- Accessors ------------------------------------------------------------
 
    public int     getCols()         { return cols; }
    public int     getRows()         { return rows; }
    public float   getResolutionM()  { return resolutionM; }
    public Instant getUpdatedAt()    { return updatedAt; }
 
    /** Returns cost at grid cell (col, row) as an unsigned integer [0, 255]. */
    public int costAt(int col, int row) {
        checkBounds(col, row);
        return Byte.toUnsignedInt(costs[row * cols + col]);
    }
 
    /** Returns cost at world position (worldX, worldY) in metres. */
    public int costAtWorld(float worldX, float worldY) {
        int col = worldToCol(worldX);
        int row = worldToRow(worldY);
        if (col < 0 || col >= cols || row < 0 || row >= rows) return COST_LETHAL;
        return costAt(col, row);
    }
 
    /** True when a rover footprint centred at (worldX, worldY) is collision-free. */
    public boolean isFree(float worldX, float worldY, float footprintM) {
        int halfCells = (int) Math.ceil((footprintM / 2f) / resolutionM);
        int cc = worldToCol(worldX), cr = worldToRow(worldY);
        for (int dr = -halfCells; dr <= halfCells; dr++) {
            for (int dc = -halfCells; dc <= halfCells; dc++) {
                int r = cr + dr, c = cc + dc;
                if (r < 0 || r >= rows || c < 0 || c >= cols) return false;
                if (costAt(c, r) >= COST_LETHAL) return false;
            }
        }
        return true;
    }
 
    // - Factory: build from a list of detection results -----------------------
 
    /**
     * Constructs a fresh costmap from the latest fused detection results.
     * For each detection, costs are inflated using a Gaussian around the
     * projected footprint; the inflation radius depends on object class and
     * the detection confidence.
     *
     * @param detections  fused detection list (should be NMS-filtered)
     * @param cols        grid width  (x axis, meters ahead)
     * @param rows        grid height (y axis, lateral)
     * @param resolution  metres per cell (e.g. 0.10f)
     */
    public static TraversabilityCostmap fromDetections(
            List<DetectionResult> detections,
            int cols, int rows, float resolution) {
 
        byte[] costs = new byte[cols * rows];  // starts at 0 (free)
 
        for (DetectionResult det : detections) {
            if (!det.has3dBox()) continue;
 
            var box = det.getBox3d();
            int cc = worldToCellX(box.cx(), cols, resolution);
            int cr = worldToCellY(box.cy(), rows, resolution);
 
            float baseCost = terrainCost(det.getTerrainClass()) * det.getConfidence();
            int   baseInt  = Math.round(baseCost * COST_LETHAL);
            float sigma    = inflationRadius(det.getTerrainClass()) / resolution;
 
            inflate(costs, cols, rows, cc, cr, baseInt, sigma);
        }
        return new Builder()
                .dimensions(cols, rows)
                .resolution(resolution)
                .origin(-(cols * resolution / 2f), 0f)
                .costs(costs)
                .build();
    }
 
    // -- Cost update / merge ----------------------------------------------------
 
    /**
     * Returns a new costmap that is the element-wise maximum of {@code this}
     * and {@code other}, preserving the highest-cost (most conservative) estimate.
     */
    public TraversabilityCostmap mergeMax(TraversabilityCostmap other) {
        if (cols != other.cols || rows != other.rows)
            throw new IllegalArgumentException("Costmap dimensions must match for merge");
        byte[] merged = new byte[costs.length];
        for (int i = 0; i < costs.length; i++) {
            merged[i] = (byte) Math.max(
                    Byte.toUnsignedInt(costs[i]),
                    Byte.toUnsignedInt(other.costs[i]));
        }
        return new Builder()
                .dimensions(cols, rows).resolution(resolutionM)
                .origin(originX, originY).costs(merged).build();
    }
 
    // -- Helpers -------------------------------------------------------------
 
    private static void inflate(byte[] costs, int cols, int rows,
                                 int cc, int cr, int baseInt, float sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        for (int dr = -radius; dr <= radius; dr++) {
            for (int dc = -radius; dc <= radius; dc++) {
                int r = cr + dr, c = cc + dc;
                if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
                double gauss = Math.exp(-(dc*dc + dr*dr) / (2 * sigma * sigma));
                int inflated = (int)(baseInt * gauss);
                int idx = r * cols + c;
                int cur = Byte.toUnsignedInt(costs[idx]);
                costs[idx] = (byte) Math.min(COST_LETHAL, Math.max(cur, inflated));
            }
        }
    }
 
    private static float terrainCost(TerrainClass tc) {
        return 1.0f - (float) tc.getTraversabilityScore();
    }
 
    private static float inflationRadius(TerrainClass tc) {
        return switch (tc) {
            case CRATER  -> 2.0f;
            case ROCK    -> 1.2f;
            case SLOPE   -> 1.5f;
            case DEBRIS  -> 1.0f;
            case UNKNOWN -> 2.5f;
            default      -> 0.3f;
        };
    }
 
    private static int worldToCellX(double worldX, int cols, float res) {
        return (int)((worldX + (cols * res / 2.0)) / res);
    }
    private static int worldToCellY(double worldY, int rows, float res) {
        return (int)(worldY / res);
    }
 
    private int worldToCol(float x) { return (int)((x - originX) / resolutionM); }
    private int worldToRow(float y) { return (int)((y - originY) / resolutionM); }
 
    private void checkBounds(int col, int row) {
        if (col < 0 || col >= cols || row < 0 || row >= rows)
            throw new IndexOutOfBoundsException(
                    "Cell (%d, %d) out of bounds [%d×%d]".formatted(col, row, cols, rows));
    }
 
    // -- Builder ----------------------------------------------------------------
 
    public static Builder builder() { return new Builder(); }
 
    public static final class Builder {
        private int     cols = 200, rows = 100;
        private float   resolutionM = 0.10f;
        private float   originX = -10f, originY = 0f;
        private byte[]  costs;
        private Instant updatedAt;
 
        public Builder dimensions(int cols, int rows) { this.cols = cols; this.rows = rows; return this; }
        public Builder resolution(float r)            { this.resolutionM = r;               return this; }
        public Builder origin(float x, float y)       { this.originX = x; this.originY = y; return this; }
        public Builder costs(byte[] c)                { this.costs = c;                     return this; }
        public Builder updatedAt(Instant t)           { this.updatedAt = t;                 return this; }
 
        public TraversabilityCostmap build() {
            if (costs == null) costs = new byte[cols * rows];
            return new TraversabilityCostmap(this);
        }
    }
}