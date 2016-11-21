import java.util.ArrayList;
import java.util.Random;
import java.io.*;


public class NeuralNet {

	public NeuralNet(int nn_neurons[])
	{
		Random rand = new Random();
		layerOfTheNeuralNet = new ArrayList<NeuralLayer>();
		for (int i = 0; i < nn_neurons.length; ++i)
			layerOfTheNeuralNet.add(
					new NeuralLayer(
							i == 0 ? 
							nn_neurons[i] : nn_neurons[i - 1], 
							nn_neurons[i], rand)
					);

		deltaWeights = new ArrayList<float[][]>();
		for (int i = 0; i < nn_neurons.length; ++i)
			deltaWeights.add(new float
						[layerOfTheNeuralNet.get(i).size()]
						[layerOfTheNeuralNet.get(i).getWeights(0).length]
					 );

		gradEx = new ArrayList<float[]>();
		for (int i =  0; i < nn_neurons.length; ++i)
			gradEx.add(new float[layerOfTheNeuralNet.get(i).size()]);
	}

	public float[] evaluate(float[] inputs)
	{
		assert (false);
		float outputs[] = new float[inputs.length];
		for(int i = 0; i < layerOfTheNeuralNet.size(); ++i ) {
			outputs = layerOfTheNeuralNet.get(i).evaluate(inputs);
			inputs = outputs;
		}

		return outputs;
	}

	private float evaluateError(float obtainedOutput[], float desiredOuput[])
	{
		float d[];
		if (desiredOuput.length != obtainedOutput.length) {
			d = NeuralLayer.addBias(desiredOuput);
		}
		else {
			d = desiredOuput;
		}
		assert(obtainedOutput.length == d.length);
		float e = 0;
		for (int i = 0; i < obtainedOutput.length; ++i)
			e += (obtainedOutput[i] - d[i]) * (obtainedOutput[i] - d[i]);
		return e;
	}

	public float evaluateQuadraticError(ArrayList<float[]> examples,
								   ArrayList<float[]> results)
	{
		assert(false);
		float e = 0;
		for (int i = 0; i < examples.size(); ++i) {
			e += evaluateError(evaluate(examples.get(i)), results.get(i));
		}
		return e;
	}

	private void evaluateGradients(float[] results)
	{
		for (int c = layerOfTheNeuralNet.size()-1; c >= 0; --c) {
			for (int i = 0; i < layerOfTheNeuralNet.get(c).size(); ++i) {
				if (c == layerOfTheNeuralNet.size()-1) {
					gradEx.get(c)[i] =
						2 * (layerOfTheNeuralNet.get(c).getOutput(i) - results[0])
						  * layerOfTheNeuralNet.get(c).getActivationDerivative(i);
				}
				else {
					float sum = 0;
					for (int k = 1; k < layerOfTheNeuralNet.get(c+1).size(); ++k)
						sum += layerOfTheNeuralNet.get(c+1).getWeight(k, i) * gradEx.get(c+1)[k];
					gradEx.get(c)[i] = layerOfTheNeuralNet.get(c).getActivationDerivative(i) * sum;
				}
			}
		}
	}

	private void resetWeightsDelta()
	{
		for (int c = 0; c < layerOfTheNeuralNet.size(); ++c) {
			for (int i = 0; i < layerOfTheNeuralNet.get(c).size(); ++i) {
				float weights[] = layerOfTheNeuralNet.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					deltaWeights.get(c)[i][j] = 0;
	        }		
		}
	}

	private void evaluateWeightsDelta()
	{
		// evaluate delta values for each weight
		for (int c = 1; c < layerOfTheNeuralNet.size(); ++c) {
			for (int i = 0; i < layerOfTheNeuralNet.get(c).size(); ++i) {
				float weights[] = layerOfTheNeuralNet.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					deltaWeights.get(c)[i][j] += gradEx.get(c)[i]
					     * layerOfTheNeuralNet.get(c-1).getOutput(j);
			}
		}
	}

	private void updateWeights(float learning_rate)
	{
		for (int c = 0; c < layerOfTheNeuralNet.size(); ++c) {
			for (int i = 0; i < layerOfTheNeuralNet.get(c).size(); ++i) {
				float weights[] = layerOfTheNeuralNet.get(c).getWeights(i);
				for (int j = 0; j < weights.length; ++j)
					layerOfTheNeuralNet.get(c).setWeight(i, j, layerOfTheNeuralNet.get(c).getWeight(i, j)
							- (learning_rate * deltaWeights.get(c)[i][j]));
	        }
		}
	}

	private void batchBackPropagation(ArrayList<float[]> examples,
									  ArrayList<float[]> results,
									  float learning_rate)
	{
		resetWeightsDelta();

		for (int l = 0; l < examples.size(); ++l) {
			evaluate(examples.get(l));
			evaluateGradients(results.get(l));
			evaluateWeightsDelta();
		}

		updateWeights(learning_rate);
	}

	public void learn(ArrayList<float[]> examples, ArrayList<float[]> results, float learning_rate)
	{
		assert(false);
		float e = Float.POSITIVE_INFINITY;
		//while (e > 0.001f) {
			batchBackPropagation(examples, results, learning_rate);
			e = evaluateQuadraticError(examples, results);
		//}
	}

	private ArrayList<NeuralLayer> layerOfTheNeuralNet;
	private ArrayList<float[][]> deltaWeights;
	private ArrayList<float[]> gradEx;


	public static void main(String[] args) {

		ArrayList<float[]> examples = new ArrayList<float[]>();
		ArrayList<float[]> category = new ArrayList<float[]>();
		for (int i = 0; i < 4; ++i) {
			examples.add(new float[2]);
			category.add(new float[1]);
		}

		// Examples
		examples.get(0)[0] = 5; examples.get(0)[1] = 1;  category.get(0)[0] = 1;
		examples.get(1)[0] = 1;  examples.get(1)[1] = 1;  category.get(1)[0] = 0;
		examples.get(2)[0] = 1;  examples.get(2)[1] = 0; category.get(2)[0] = 1;
		examples.get(3)[0] = 6; examples.get(3)[1] = 1; category.get(3)[0] = 0;

		ListOfExamples trainExamples = new ListOfExamples();
		trainExamples.ReadInExamplesFromFile("red-wine-quality-train.txt");

		ArrayList<float[]> examplesWine = new ArrayList<float[]>();
		ArrayList<float[]> categoryWine = new ArrayList<float[]>();
		for (Example trainExample : trainExamples) {
			float[] features = new float[trainExample.size()];
			for (int i = 0; i < trainExample.size(); i++) {
				features[i] = trainExamples.getFeatures()[i].getFirstValue().equalsIgnoreCase(trainExample.get(i)) ? 0 : 1;
			}
			float[] outcome = new float[1];
			outcome[0]= trainExamples.getOutputLabel().getFirstValue().equalsIgnoreCase(trainExample.getLabel()) ? 0 : 1;
			examplesWine.add(features);
			categoryWine.add(outcome);
		}

		int nn_neurons[] = { 20, 5, 1 };

		NeuralNet neuralNet = new NeuralNet(nn_neurons);

		try {
			PrintWriter fout = new PrintWriter(new FileWriter("code.py"));

			for (int i = 0; i < 20; ++i) {
				neuralNet.learn(examplesWine, categoryWine, 0.3f);
				//System.out.println("Learn " + i + " Done");

				float error = neuralNet.evaluateQuadraticError(examplesWine, categoryWine);
				System.out.println(i + " -> error : " + error);
			}

			StringBuilder methodDefinition = new StringBuilder();
			methodDefinition.append("def NN(");
			//fout.write("def NN(");
			for (int i = 1; i < neuralNet.layerOfTheNeuralNet.get(0).getNeurons().size(); i++) {
				if(i==(neuralNet.layerOfTheNeuralNet.get(0).getNeurons().size()-1))
				{
					methodDefinition.append("n0" + (i));
				}else{
					methodDefinition.append("n0" + (i) + ", ");
				}
			}
			methodDefinition.append("):\n");
			fout.write(methodDefinition.toString());
			String last = "";
			for(int layer = 1; layer< neuralNet.layerOfTheNeuralNet.size(); layer ++){
				for (int neuron = 1; neuron < neuralNet.layerOfTheNeuralNet.get(layer).getNeurons().size(); neuron++) {
					float[] weights = neuralNet.layerOfTheNeuralNet.get(layer).getWeights(neuron);
					StringBuilder stringBuilder = new StringBuilder();
					stringBuilder.append("n" + layer + neuron + "_1 = ");
					for (int i = 0; i < weights.length; i++) {
						if(i == 0){
							stringBuilder.append(weights[i] + " * 1 + ");
						}else{
							stringBuilder.append(weights[i] + " * n" + (layer-1) +"" + (i) + " + ");
						}
					}
					String s = stringBuilder.toString();
					//System.out.println(s.substring(0,s.length() - 2));
					fout.write("\t"+s.substring(0,s.length() - 2));
					fout.println();
					fout.write("\tif "+s.substring(0,3)+"_1 < 0:\n");
					fout.write("\t\t" + s.substring(0,3) + "=0\n");
					fout.write("\telse:\n");
					fout.write("\t\t" + s.substring(0,3) + "="+s.substring(0,3)+ "_1\n");
					last = s;
				}
			}
			fout.write("\treturn " + last.substring(0,3));
			fout.close();
			System.out.println("Done");
		} catch (IOException e){
			e.printStackTrace();
		}
	}

}
