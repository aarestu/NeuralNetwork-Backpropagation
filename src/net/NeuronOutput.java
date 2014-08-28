
package net;

/**
 *
 * @author aarestu
 */
public class NeuronOutput extends Neuron {
    
    protected double m_galat;
    
    public NeuronOutput(int myIndex) {
        super(myIndex);
    }
    
    @Override
    public void updateWeight(double galat, int i) {
        System.out.println("output neuron tidak mempunyai bobot koneksi");
    }
    
    @Override
    public void hitungGalat(double targetVal) {
        double delta = targetVal - this.getOutputVal();
        this.m_galat = delta * TransferFunction.transferTurunan(this.m_outputVal);
    }
    
    @Override
    public double getGalat() {
        return this.m_galat;
    }
}
