
package net;

/**
 *
 * @author aarestu
 */
public class NeuronHidden extends Neuron {
    
    protected double m_galat;
    protected Connection[] m_outputWeights;
    
    public NeuronHidden(int numOutput, int myIndex) {
        super(myIndex);
        
        this.m_outputWeights = new Connection[numOutput];
        for (int i = 0; i < numOutput; i++) {
            this.m_outputWeights[i] = new Connection();
            this.m_outputWeights[i].setWeight(Neuron.weightAcak());
        }
    }
    
    @Override
    public void hitungGalat(Layer layerSelanjutnya) {
        double sum = 0.0;

        for (int i = 0; i < layerSelanjutnya.getJumlahNeuron() - 1; i++) {
            sum += this.m_outputWeights[i].getWeight()
                    * layerSelanjutnya.neuron[i].getGalat();
        }

        this.m_galat = sum * TransferFunction.transferTurunan(this.getOutputVal());
    }
    
    @Override
    public Connection getOutputWeight(int index) {
        return this.m_outputWeights[index];
    }
    
    @Override
    public double getGalat() {
        return this.m_galat;
    }
}
