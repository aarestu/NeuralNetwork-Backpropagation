
package net;

/**
 *
 * @author aarestu
 */
public class Layer {
    public Neuron[] neuron;
    
    public Layer(Neuron[] neuron){
        this.neuron = neuron;
    }
    
    public int getJumlahNeuron(){
        return neuron.length;
    }
    
    public Neuron getIndexNeuron(int i){
          return neuron[i];
    }
}
