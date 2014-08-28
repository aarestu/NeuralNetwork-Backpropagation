
package net;

public class Net {

    public static double learningrate = 0.25;
    public static double momentum = 0.5;
    
    private Layer[] m_layer;
    private double m_error;
    public double sum_global_error;

    public Net() {
        
    }

    public void setTopology(int[] topology) {
        int numLayers = topology.length;
        if(numLayers < 3){
            System.out.println("minimal 3 layer : input, hidden, output");
            return;
        }

        this.m_layer = new Layer[numLayers];
        
        // Inisialisasi neuron dan bias untuk input layer
        this.m_layer[0] = new Layer(new NeuronInput[topology[0] + 1]);
        
        int numOutput = topology[1];
        for (int neuronNum = 0; neuronNum <= topology[0]; neuronNum++) {
            this.m_layer[0].neuron[neuronNum] = new NeuronInput(numOutput, neuronNum);
        }
        
        // neuron terakhir berperan sebagai bias, isi output = 1
        this.m_layer[0].neuron[topology[0]].setOutputVal(1);
        
        // Inisialisasi hidden layer
        for (int layerNum = 1; layerNum < numLayers - 1; layerNum++) {
            
            // Buat neuron dan bias untuk setiap layer
            this.m_layer[layerNum] = new Layer(new NeuronHidden[topology[layerNum] + 1]);

            numOutput = topology[layerNum + 1];

            for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
                this.m_layer[layerNum].neuron[neuronNum] = new NeuronHidden(numOutput, neuronNum);
            }

            //neuron terakhir berperan sebagai bias, isi output = 1
            this.m_layer[layerNum].neuron[topology[layerNum]].setOutputVal(1);
        }
        
        // Inisialisasi output layer
        // Inisialisasi neuron dan bias untuk input layer
        this.m_layer[numLayers - 1] = new Layer(new NeuronOutput[topology[numLayers - 1] + 1]);
        
        for (int neuronNum = 0; neuronNum <= topology[numLayers - 1]; neuronNum++) {
            this.m_layer[numLayers - 1].neuron[neuronNum] = new NeuronOutput(neuronNum);
        }
        this.m_layer[numLayers - 1].neuron[topology[numLayers - 1]].setOutputVal(1);
    }

    public void backProp(double[] targetVal) {
        if (this.m_layer == null || this.m_layer.length < 2) {
            System.out.println("Arsitektur NN belum terbentuk dengan benar");
            return;
        }

        int outputNum = this.m_layer[this.m_layer.length - 1].getJumlahNeuron() - 1;

        if (targetVal.length != outputNum) {
            System.out.println("banyak target tidak sama dengan banyak neuron di output layer");
            return;
        }

        //hitung error local
        this.m_error = 0.0;
        for (int i = 0; i < outputNum; i++) {
            double delta = targetVal[i]
                    - this.m_layer[this.m_layer.length - 1].neuron[i].getOutputVal();
            this.m_error = delta * delta;
            this.sum_global_error += this.m_error;
        }
        this.m_error = this.m_error / outputNum; // Mean Squared Error (MSE)
        this.m_error = Math.sqrt(this.m_error); // RMS

        //hitung galat output layer
        for (int i = 0; i < outputNum; i++) {
            this.m_layer[this.m_layer.length - 1].neuron[i].hitungGalat(targetVal[i]);
        }

        //hitung galat hidden layer
        for (int layerNum = this.m_layer.length - 2; layerNum >= 0; layerNum--) {
            Layer layerSelanjutnya = this.m_layer[layerNum + 1];

            for (int i = 0; i < this.m_layer[layerNum].neuron.length; i++) {
                this.m_layer[layerNum].neuron[i].hitungGalat(layerSelanjutnya);
            }
        }

        //update bobot koneksi
        for (int layerNum = this.m_layer.length - 1; layerNum > 0; layerNum--) {
            for (int n = 0; n < this.m_layer[layerNum].neuron.length - 1; n++) {
                double galat = this.m_layer[layerNum].neuron[n].getGalat();

                for (int i = 0; i < this.m_layer[layerNum - 1].neuron.length; i++) {
                    this.m_layer[layerNum - 1].neuron[i].updateWeight(galat, n);
                }

            }
        }
    }

    public void feedForward(double[] inputVal) {
        if (this.m_layer == null || this.m_layer.length < 2) {
            System.out.println("Arsitektur NN belum terbentuk dengan benar");
            return;
        }

        if (inputVal.length != this.m_layer[0].neuron.length - 1) {
            System.out.println("banyak input tidak sama dengan banyak neuron di input layer");
            return;
        }

        //inisialisasi input layer
        for (int i = 0; i < inputVal.length; i++) {
            this.m_layer[0].neuron[i].setOutputVal(inputVal[i]);
        }

        //feed forwared propagate
        for (int layerNum = 1; layerNum < this.m_layer.length; layerNum++) {
            Layer layerSebelumnya = this.m_layer[layerNum - 1];
            for (int n = 0; n < this.m_layer[layerNum].neuron.length - 1; n++) {
                this.m_layer[layerNum].neuron[n].feedForward(layerSebelumnya);
            }
        }
    }

    public double getError() {
        return this.m_error;
    }

    public double[] getHasil() {
        int outputNum = this.m_layer[this.m_layer.length - 1].neuron.length - 1;
        double[] hasil = new double[outputNum];
        for (int i = 0; i < outputNum; i++) {
            hasil[i] = 0;
            hasil[i] = this.m_layer[this.m_layer.length - 1].neuron[i].getOutputVal();

        }
        return hasil;
    }
}
