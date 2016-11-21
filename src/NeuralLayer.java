import java.util.ArrayList;


public class NeuralLayer {

	private int numberOfNeuronsInThisLayer, numberOfNeuronsInPrevLayer;
	private ArrayList<Neuron> neurons;
	private float outputs[];

	public NeuralLayer(int prevNneurons, int nNeurons, java.util.Random rand)
	{
		numberOfNeuronsInThisLayer = nNeurons + 1;
		numberOfNeuronsInPrevLayer = prevNneurons + 1;
		neurons = new ArrayList<Neuron>();
		outputs = new float[numberOfNeuronsInThisLayer];

		for (int i = 0; i < numberOfNeuronsInThisLayer; ++i)
			neurons.add(new Neuron(numberOfNeuronsInPrevLayer, rand));
	}

	public static float[] addBias(float[] in)
	{
		float out[] = new float[in.length + 1];
		for (int i = 0; i < in.length; ++i)
			out[i + 1] = in[i];
		out[0] = 1.0f;
		return out;
	}

	public float[] evaluate(float in[])
	{
		float inputs[];
		if (in.length != getWeights(0).length)
			inputs = addBias(in);
		else
			inputs = in;

		assert(getWeights(0).length == inputs.length);
		for (int i = 1; i < numberOfNeuronsInThisLayer; ++i) {
			outputs[i] = neurons.get(i).activate(inputs);
		}
		outputs[0] = 1.0f;

		return outputs;
	}

	public int size() { return numberOfNeuronsInThisLayer; }
	public float getOutput(int i) { return outputs[i]; }
	public float getActivationDerivative(int i) { return neurons.get(i).getActivationDerivative(); }
	public float[] getWeights(int i) { return neurons.get(i).getSynapticWeights(); }
	public float getWeight(int i, int j) { return neurons.get(i).getSynapticWeight(j); }
	public void setWeight(int i, int j, float v) { neurons.get(i).setSynapticWeight(j, v); }

	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}
}
