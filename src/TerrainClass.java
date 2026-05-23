/**
 * @author Hayden Zhang
 * Date: 5/23/26
 * NASA Space Grant Fellowship V.I.S.O.R. Project
 * Terrain classification labels for the MarsTerrain-12k dataset.
 * Each class carries a traversability score (0.0 = impassable, 1.0 = fully safe)
 * and an alert priority used by the path planners costmap generator. 
 */


public enum TerrainClass {

     
    CLEAR_PATH ("Clear / Traversable", 1.00, AlertPriority.NONE),
    ROCK       ("Rock / Boulder", 0.10, AlertPriority.HIGH),
    CRATER     ("Crater / Depression", 0.05, AlertPriority.CRITICAL),
    SLOPE      ("Slope / Incline >15°", 0.30, AlertPriority.MEDIUM),
    DEBRIS     ("Uknown / Occluded", 0.00, AlertPriority.CRITICAL);

    private final String displayName;
    private final double traversabilityScore;
    private final AlertPriority alertPriority; 

    TerrainClass(String displayName, double traversabilityScore, AlertPriority alertPriority) {
        this.displayName = displayName;
        this.traversabilityScore = traversabilityScore;
        this.alertPriority = alertPriority; 

    }

    public String getDisplayName() { return displayName; }
    public double getraversabilityScore() { return traversabilityScore; }
    public AlertPriority getAlertPriority() { return alertPriority; }

    /**
     * Returns true when the rover cannot proceed without human command override. 
     */
    
    public boolean isHazardous() {
        return alertPriority == AlertPriority.HIGH || alertPriority == AlertPriority.CRITICAL;
    }

    public enum AlertPriority { NONE, Low, MEDIUM, HIGH, CRITICAL }
}
