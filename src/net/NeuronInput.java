
package net;

/**
 *
 * @author aarestu
 */
public class NeuronInput extends Neuron {
    
    protected Connection[] m_outputWeights;
    public NeuronInput(int numOutput, int myIndex) {
        super(myIndex);
        
        this.m_outputWeights = new Connection[numOutput];
        for (int i = 0; i < numOutput; i++) {
            this.m_outputWeights[i] = new Connection();
            this.m_outputWeights[i].setWeight(Neuron.weightAcak());
        }
    }
    
    @Override
    public Connection getOutputWeight(int index) {
        return this.m_outputWeights[index];
    }
}
