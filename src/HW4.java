import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class HW4
{
    private static int epoch = 1000;
    private static int printDetailsAtEveryEpoch = 50;
    private static double learningRate = 0.1d;
    private static TreeLogger logger = new TreeLogger(false);
    public static void main(String[] args)
    {
        if (args.length != 3)
        {
            System.err.println("You must call BuildAndTestDecisionTree as " +
                    "follows:\n\njava BuildAndTestDecisionTree " +
                    "<trainsetFilename> <tunesetFileName> <testsetFilename>\n");
            System.exit(1);
        }

        // Read in the file names.
        String trainset = args[0];
        String tuneSet = args[1];
        String testset  = args[2];

        ListOfExamples trainExamples = new ListOfExamples();
        ListOfExamples testExamples  = new ListOfExamples();
        ListOfExamples tuneExamples = new ListOfExamples();

        if (!trainExamples.ReadInExamplesFromFile(trainset) ||
                !testExamples.ReadInExamplesFromFile(testset) ||
                !tuneExamples.ReadInExamplesFromFile(tuneSet))
        {
            System.err.println("Something went wrong reading the datasets ... " +
                    "giving up.");
            System.exit(1);
        }
        else
        {
            double[] weights = new double[trainExamples.getFeatures().length + 1];

            double maxTuneSetAccuracy = Double.MIN_VALUE;
            int maxTuneSerAccuracyWasObtainedAt = 0;
            double testSetAccuracyAtMaxTuneSetAccuracy = 0;

            double maxTestSetAccuracy = Double.MIN_VALUE;
            int maxTestSetAccuracyWasObtainedAt = 0;
            double tuneSetAccuracyAtMaxTestSetAccuracy = 0;

            OrderExamples orderExamples = new RandomOrderExamples();

            for (int ep = 1; ep <= epoch; ep++) {
                for (Example trainExample : orderExamples.getExamples(trainExamples)) {
                    //logger.log("----------------------------------------");
                    //logger.log(Arrays.toString(weights));
                    int[] feature = getFeaturesAsZeroAndOnesInArray(trainExample, trainExamples);
                    int expectedOutput = convertoOneAndZero(trainExample, trainExamples);
                    int predictedOutput = predictOutput(weights, feature);
                    //logger.log("Pre correction = " + predictOutput(weights, feature) + ":Expected =" + expectedOutput);
                    if(expectedOutput != predictedOutput){
                        weights  = correctTheWeights(weights, feature, expectedOutput , predictedOutput);
                    }
                    //logger.log(Arrays.toString(feature));
                    //logger.log(Arrays.toString(weights));
                    //logger.log("Post correction = " + predictOutput(weights, feature) + ":Expected =" + expectedOutput);
                    //logger.log("-----------------------------------------");
                }
                double tuneSetAccuracy = getAccuray(weights, tuneExamples);
                double testSetAccuracy = getAccuray(weights, testExamples);
                if(tuneSetAccuracy > maxTuneSetAccuracy){
                    maxTuneSetAccuracy = tuneSetAccuracy;
                    maxTuneSerAccuracyWasObtainedAt = ep;
                    testSetAccuracyAtMaxTuneSetAccuracy = testSetAccuracy;
                }
                if(testSetAccuracy > maxTestSetAccuracy){
                    maxTestSetAccuracy = testSetAccuracy;
                    maxTestSetAccuracyWasObtainedAt = ep;
                    tuneSetAccuracyAtMaxTestSetAccuracy = tuneSetAccuracy;
                }

                if(ep % printDetailsAtEveryEpoch == 0){
                    double trainSetAccuracy = getAccuray(weights, trainExamples);
                    System.out.format("Epoch %d: train = %.2f%% tune = %.2f%% test = %.2f%% \n",ep,trainSetAccuracy, tuneSetAccuracy, testSetAccuracy);
                    //System.out.println("Epoch "+ ep +": train = "+ trainSetAccuracy+ "% tune = "+ tuneSetAccuracy +"" +
                    //        "% test = "+ testSetAccuracy +" %");
                }
            }
            System.out.format("The tune set was highest (%.2f%% accuracy) at Epoch %d. Test set = %.2f%% here.\n",maxTuneSetAccuracy,
                    maxTuneSerAccuracyWasObtainedAt, testSetAccuracyAtMaxTuneSetAccuracy);
            System.out.format("The test set was highest (%.2f%% accuracy) at Epoch %d. Tune set = %.2f%% here.\n",maxTestSetAccuracy,
                    maxTestSetAccuracyWasObtainedAt, tuneSetAccuracyAtMaxTestSetAccuracy);

            for (int i = 0; i < trainExamples.getFeatures().length; i++) {
                System.out.format("Wgt = %.2f %s\n", weights[i], trainExamples.getFeatureName(i));
            }

        }
    }



    private static double getAccuray(double[] weights, ListOfExamples examples) {
        double correctlyPredicted = 0;
        double incorrectlyPredicted = 0;
        for (Example example : examples) {
            int[] feature = getFeaturesAsZeroAndOnesInArray(example, examples);
            int output = predictOutput(weights, getFeaturesAsZeroAndOnesInArray(example, examples));
            int expectedOutput = convertoOneAndZero(example, examples);
            //logger.log("Accuracy ouput = " + predictOutput(weights, feature) + ":Expected =" + expectedOutput);
            if(expectedOutput == output){
                correctlyPredicted++;
            }else{
                incorrectlyPredicted++;
            }
        }
        return ( correctlyPredicted * 100 / ( correctlyPredicted + incorrectlyPredicted));
    }

    private static double[] correctTheWeights(double[] weights, int[] feature, int expectedOutput, int predictedOutput) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += ( learningRate * (expectedOutput - predictedOutput) * feature[i]);
        }
        return weights;
    }

    private static int convertoOneAndZero(Example trainExample, ListOfExamples trainExamples) {
        return trainExamples.getOutputLabel().getFirstValue().equalsIgnoreCase(trainExample.getLabel()) ? 0 : 1;
    }

    private static int predictOutput(double[] weights, int[] feature) {
        double sum = 0;
        for (int i = 0; i < feature.length; i++) {
            sum += weights[i]*feature[i];
        }
        int prediction = applyActivationFuntion(sum);
        return prediction;
    }

    private static int applyActivationFuntion(double sum) {
        return sum >= 0.0d ? 1 : 0;
    }

    private static int[] getFeaturesAsZeroAndOnesInArray(Example trainExample, ListOfExamples trainExamples) {
        int[] features = new int[trainExample.size() + 1];
        for(int i = 0 ;i < trainExample.size() ; i++){
            BinaryFeature binaryFeature = trainExamples.getFeatures()[i];
            if(binaryFeature.getFirstValue().equalsIgnoreCase(trainExample.get(i))){
                features[i] = 0;
            }else{
                features[i] = 1;
            }
        }
        features[features.length - 1] = -1;
        return features;
    }
}

// This class, an extension of ArrayList, holds an individual example.
// The new method PrintFeatures() can be used to
// display the contents of the example.
// The items in the ArrayList are the feature values.
class Example extends ArrayList<String>
{
    // The name of this example.
    private String name;

    // The output label of this example.
    private String label;

    // The data set in which this is one example.
    private ListOfExamples parent;

    // Constructor which stores the dataset which the example belongs to.
    public Example(ListOfExamples parent) {
        this.parent = parent;
    }

    // Print out this example in human-readable form.
    public void PrintFeatures()
    {
        System.out.print("Example " + name + ",  label = " + label + "\n");
        for (int i = 0; i < parent.getNumberOfFeatures(); i++)
        {
            System.out.print("     " + parent.getFeatureName(i)
                    + " = " +  this.get(i) + "\n");
        }
    }

    // Adds a feature value to the example.
    public void addFeatureValue(String value) {
        this.add(value);
    }

    // Accessor methods.
    public String getName() {
        return name;
    }

    public String getLabel() {
        return label;
    }

    // Mutator methods.
    public void setName(String name) {
        this.name = name;
    }

    public void setLabel(String label) {
        this.label = label;
    }

}

/* This class holds all of our examples from one dataset
   (train OR test, not BOTH).  It extends the ArrayList class.
   Be sure you're not confused.  We're using TWO types of ArrayLists.
   An Example is an ArrayList of feature values, while a ListOfExamples is
   an ArrayList of examples. Also, there is one ListOfExamples for the
   TRAINING SET and one for the TESTING SET.
*/
class ListOfExamples extends ArrayList<Example>
{
    // The name of the dataset.
    private String nameOfDataset = "";

    // The number of features per example in the dataset.
    private int numFeatures = -1;

    // An array of the parsed features in the data.
    private BinaryFeature[] features;

    // A binary feature representing the output label of the dataset.
    private BinaryFeature outputLabel;

    // The number of examples in the dataset.
    private int numExamples = -1;

    public ListOfExamples() {}

    public ListOfExamples(List<Example> withFirstValue, BinaryFeature outputLabel, BinaryFeature[] features) {
        this.addAll(withFirstValue);
        this.setOutputLabel(outputLabel);
        this.features = features;
    }

    private void setOutputLabel(BinaryFeature outputLabel){
        this.outputLabel = outputLabel;
    }

    // Print out a high-level description of the dataset including its features.
    public void DescribeDataset()
    {
        System.out.println("Dataset '" + nameOfDataset + "' contains "
                + numExamples + " examples, each with "
                + numFeatures + " features.");
        System.out.println("Valid category labels: "
                + outputLabel.getFirstValue() + ", "
                + outputLabel.getSecondValue());
        System.out.println("The feature names (with their possible values) are:");
        for (int i = 0; i < numFeatures; i++)
        {
            BinaryFeature f = features[i];
            System.out.println("   " + f.getName() + " (" + f.getFirstValue() +
                    " or " + f.getSecondValue() + ")");
        }
        System.out.println();
    }

    public BinaryFeature getOutputLabel() {
        return outputLabel;
    }

    // Print out ALL the examples.
    public void PrintAllExamples()
    {
        System.out.println("List of Examples\n================");
        for (int i = 0; i < size(); i++)
        {
            Example thisExample = this.get(i);
            thisExample.PrintFeatures();
        }
    }

    // Print out the SPECIFIED example.
    public void PrintThisExample(int i)
    {
        Example thisExample = this.get(i);
        thisExample.PrintFeatures();
    }

    // Returns the number of features in the data.
    public int getNumberOfFeatures() {
        return numFeatures;
    }

    // Returns the name of the ith feature.
    public String getFeatureName(int i) {
        return features[i].getName();
    }

    public BinaryFeature[] getFeatures() {
        return features;
    }

    // Takes the name of an input file and attempts to open it for parsing.
    // If it is successful, it reads the dataset into its internal structures.
    // Returns true if the read was successful.
    public boolean ReadInExamplesFromFile(String dataFile) {
        nameOfDataset = dataFile;

        // Try creating a scanner to read the input file.
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(dataFile));
        } catch(FileNotFoundException e) {
            return false;
        }

        // If the file was successfully opened, read the file
        this.parse(fileScanner);
        return true;
    }

    /**
     * Does the actual parsing work. We assume that the file is in proper format.
     *
     * @param fileScanner a Scanner which has been successfully opened to read
     * the dataset file
     */
    public void parse(Scanner fileScanner) {
        // Read the number of features per example.
        numFeatures = Integer.parseInt(parseSingleToken(fileScanner));

        // Parse the features from the file.
        parseFeatures(fileScanner);

        // Read the two possible output label values.
        String labelName = "output";
        String firstValue = parseSingleToken(fileScanner);
        String secondValue = parseSingleToken(fileScanner);
        outputLabel = new BinaryFeature(labelName, firstValue, secondValue);

        // Read the number of examples from the file.
        numExamples = Integer.parseInt(parseSingleToken(fileScanner));

        parseExamples(fileScanner);
    }

    /**
     * Returns the first token encountered on a significant line in the file.
     *
     * @param fileScanner a Scanner used to read the file.
     */
    private String parseSingleToken(Scanner fileScanner) {
        String line = findSignificantLine(fileScanner);

        // Once we find a significant line, parse the first token on the
        // line and return it.
        Scanner lineScanner = new Scanner(line);
        return lineScanner.next();
    }

    /**
     * Reads in the feature metadata from the file.
     *
     * @param fileScanner a Scanner used to read the file.
     */
    private void parseFeatures(Scanner fileScanner) {
        // Initialize the array of features to fill.
        features = new BinaryFeature[numFeatures];

        for(int i = 0; i < numFeatures; i++) {
            String line = findSignificantLine(fileScanner);

            // Once we find a significant line, read the feature description
            // from it.
            Scanner lineScanner = new Scanner(line);
            String name = lineScanner.next();
            String dash = lineScanner.next();  // Skip the dash in the file.
            String firstValue = lineScanner.next();
            String secondValue = lineScanner.next();
            features[i] = new BinaryFeature(name, firstValue, secondValue);
        }
    }

    private void parseExamples(Scanner fileScanner) {
        // Parse the expected number of examples.
        for(int i = 0; i < numExamples; i++) {
            String line = findSignificantLine(fileScanner);
            Scanner lineScanner = new Scanner(line);

            // Parse a new example from the file.
            Example ex = new Example(this);

            String name = lineScanner.next();
            ex.setName(name);

            String label = lineScanner.next();
            ex.setLabel(label);

            // Iterate through the features and increment the count for any feature
            // that has the first possible value.
            for(int j = 0; j < numFeatures; j++) {
                String feature = lineScanner.next();
                ex.addFeatureValue(feature);
            }

            // Add this example to the list.
            this.add(ex);
        }
    }

    /**
     * Returns the next line in the file which is significant (i.e. is not
     * all whitespace or a comment.
     *
     * @param fileScanner a Scanner used to read the file
     */
    private String findSignificantLine(Scanner fileScanner) {
        // Keep scanning lines until we find a significant one.
        while(fileScanner.hasNextLine()) {
            String line = fileScanner.nextLine().trim();
            if (isLineSignificant(line)) {
                return line;
            }
        }

        // If the file is in proper format, this should never happen.
        System.err.println("Unexpected problem in findSignificantLine.");

        return null;
    }

    /**
     * Returns whether the given line is significant (i.e., not blank or a
     * comment). The line should be trimmed before calling this.
     *
     * @param line the line to check
     */
    private boolean isLineSignificant(String line) {
        // Blank lines are not significant.
        if(line.length() == 0) {
            return false;
        }

        // Lines which have consecutive forward slashes as their first two
        // characters are comments and are not significant.
        if(line.length() > 2 && line.substring(0,2).equals("//")) {
            return false;
        }

        return true;
    }

    public int getFeatureIndex(BinaryFeature binaryFeature) {
        for (int i = 0; i < this.getFeatures().length; i++) {
            if(features[i].getName().equals(binaryFeature.getName())) return i;
        }
        return -1;
    }

    public int getFeatureIndex(String feature) {
        for (int i = 0; i < this.getFeatures().length; i++) {
            if(features[i].getName().equals(feature)) return i;
        }
        return -1;
    }
}

/**
 * Represents a single binary feature with two String values.
 */
class BinaryFeature {
    private String name;
    private String firstValue;
    private String secondValue;

    public BinaryFeature(String name, String first, String second) {
        this.name = name;
        firstValue = first;
        secondValue = second;
    }

    public String getName() {
        return name;
    }

    public String getFirstValue() {
        return firstValue;
    }

    public String getSecondValue() {
        return secondValue;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BinaryFeature that = (BinaryFeature) o;

        return name.equals(that.name);

    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return "BinaryFeature{" +
                "name='" + name + '\'' +
                '}';
    }
}

class TreeLogger
{
    private boolean isDebug;

    public TreeLogger(boolean isDebug){
        this.isDebug = isDebug;
    }

    public void log(String message){
        if(isDebug){
            System.out.println(message);
        }
    }
}

abstract class OrderExamples
{
    abstract List<Example> getExamples(ListOfExamples examples);
}

class RandomOrderExamples extends OrderExamples{

    public RandomOrderExamples(){
        System.out.println("Randomly choosing training examples for epochs");
    }

    @Override
    public List<Example> getExamples(ListOfExamples example) {
        List<Example> toReturn = new ArrayList();
        for(int i = 0; i< example.size() ; i++){
            Random random = new Random();
            toReturn.add(example.get(random.nextInt(example.size())));
        }
        return toReturn;
    }
}

class PreservedOrderExamples extends OrderExamples{

    public PreservedOrderExamples(){
        System.out.println("Returns examples with insertion order");
    }
    @Override
    public List<Example> getExamples(ListOfExamples examples) {
        return examples;
    }
}

