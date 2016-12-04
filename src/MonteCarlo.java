import org.python.core.PyDictionary;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MonteCarlo {

    public static void main(String[] args) throws IOException {
        new MonteCarlo().process("/Users/mukilanashokvijaya/IdeaProjects/NeuralNet/src/diabeties.txt");
    }

    public void process(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line = null;
        int[] arrayMaxRange = null;
        ArrayList<float[]> examplesDiabeties = new ArrayList<float[]>();
        ArrayList<float[]> categoryDiabeties = new ArrayList<float[]>();

        List<Integer> categoryRange = new ArrayList<>();
        while((line = reader.readLine()) != null){
            String[] split = line.split(",");
            float[] examples = new float[split.length  -1];
            float[] category = new float[1];
            categoryRange.add(split[split.length -1 ].equals("positive") ? 1 : 0);
            int length = split.length;
            if(arrayMaxRange == null){
                arrayMaxRange = new int[length - 1];
            }
            for (int i = 0; i < split.length - 1; i++) {
                arrayMaxRange[i] = (int) Math.max(arrayMaxRange[i],Double.valueOf(split[i]));
                examples[i] = Float.valueOf(split[i]);
            }
            category[0] = split[split.length - 1].equalsIgnoreCase("positive") ? 1 : 0;
            examplesDiabeties.add(examples);
            categoryDiabeties.add(category);
        }

        int count = 100;

        for(int i = 0 ; i< count ; i++){
            int[] sample = this.sample(arrayMaxRange);
            PythonInterpreter interpreter = new PythonInterpreter();
            interpreter.execfile("code.py");
            PyObject eval = interpreter.eval(String.format("NN(%d,%d,%d,%d,%d,%d,%d,%d)",sample[0],sample[1],sample[2]
                    ,sample[3],sample[4],sample[5],sample[6],sample[7]));
            System.out.println(String.format("NN(%d,%d,%d,%d,%d,%d,%d,%d)=" + eval,sample[0],sample[1],sample[2]
                    ,sample[3],sample[4],sample[5],sample[6],sample[7]));
        }
        //1,79,60,42,48,43.5,0.678,23,negative

        PythonInterpreter interpreter = new PythonInterpreter();
        interpreter.execfile("code.py");
        PyObject eval = interpreter.eval(String.format("NN(%f,%f,%f,%f,%f,%f,%f,%f)",1d,79d,60d,42d,48d,43.5d,0.678d,23d));
        System.out.println(String.format("NN(%f,%f,%f,%f,%f,%f,%f,%f)=" + eval,1d,79d,60d,42d,48d,43.5d,0.678d,23d));
    }

    private int[] sample(int[] arrayMaxRange) {
        int[] sample = new int[arrayMaxRange.length];
        Random random = new Random();
        for (int i = 0; i < sample.length; i++) {
            int i1 = random.nextInt(arrayMaxRange[i]);
            sample[i] = i1;
        }
        return sample;
    }
}
