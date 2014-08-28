
package net;

/**
 *
 * @author aarestu
 */
public class Neuron {

    protected double m_outputVal;
    protected int m_myIndex;

    public Neuron(int myIndex) {
        this.m_myIndex = myIndex;
    }

    public void updateWeight(double galat, int i) {

        double deltaWeightDulu = this.getOutputWeight(i).getDeltaweight();

        double deltaWeightBaru =
                Net.learningrate
                * this.getOutputVal()
                * galat
                + Net.momentum
                * deltaWeightDulu;

        this.getOutputWeight(i).setDeltaweight(deltaWeightBaru);
        deltaWeightBaru += this.getOutputWeight(i).getWeight();
        this.getOutputWeight(i).setWeight(deltaWeightBaru);


    }

    public void hitungGalat(Layer layerSelanjutnya){}
    public void hitungGalat(double targetVal) {}

    public void feedForward(Layer layerSebelumnya) {
        double sum = 0.0;

        for (int i = 0; i < layerSebelumnya.getJumlahNeuron(); i++) {
            sum += layerSebelumnya.neuron[i].getOutputVal()
                    * layerSebelumnya.neuron[i].getOutputWeight(this.m_myIndex).getWeight();
        }

        this.m_outputVal = TransferFunction.transfer(sum);
    }

    protected static double weightAcak() {
        //nilai acak [-1 sampai 1]
        return Math.round(Math.random()) * 2 - 1;
    }

    public double getOutputVal() {
        return this.m_outputVal;
    }

    public void setOutputVal(double m_outputVal) {
        this.m_outputVal = m_outputVal;
    }

    public double getGalat() {
        return -1;
    }

    public Connection getOutputWeight(int index) {
        return null;
    }
}
