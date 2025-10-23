import java.util.List;

// constructor 
public class Node() {
    
    // instance variables 
    private double latitude;
    private double longitude; 
    private int ID;
    private final Graph graphLoc;

    // constructor 
    public Node(double latitude, double longitude, int ID, Graph graph) {
        self.latitude = latitude; 
        self.longitude = longitude; 
        self.ID = ID; 
        self.graphLoc = graph; 
    }
    
    // get and set latitude
    private double getLatitude() {
        return latitude;
    }

    private void setLatitude(int lat) {
        latitude = lat
    }

    // get and set longitude

    private double getLongitude() {
        return longitude;
    }

    private void setLongitude(int longi) {
        longitude = longi;
    }

    //just get ID (no setter because we don't want the ID to change)

    private int getID() {
        return ID
    }

    public List getNeighbors() {
        return graphLoc.getNeighbors(ID)
    }

    @Override
    public String toString() {
        return "Node {id =" + id + ", latitude=" + latitude + ", longitude=" +longitude + "}"; 
    }
}