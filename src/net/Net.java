package net;

public class Net {

    static double learningrate = 0.25;
    static double momentum = 0.5;
    private Layer[] m_layer;
    private double m_error;
    public double sum_global_error;

    public Net() {
    }

    public void setTopology(int[] topology) {
        int numLayers = topology.length;
        this.m_layer = new Layer[numLayers];

        for (int layerNum = 0; layerNum < numLayers; layerNum++) {
            this.m_layer[layerNum] = new Layer();

            // Buat neuron dan bias untuk setiap layer
            this.m_layer[layerNum].neuron = new Neuron[topology[layerNum] + 1];

            int numOutput = (layerNum == numLayers - 1) ? 0 : topology[layerNum + 1];

            for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
                this.m_layer[layerNum].neuron[neuronNum] = new Neuron(numOutput, neuronNum);
            }

            //neuron terakhir berperan sebagai bias, isi output = 1
            this.m_layer[layerNum].neuron[topology[layerNum]].setOutputVal(1);
        }

    }

    public void backProp(double[] targetVal) {
        if (this.m_layer == null || this.m_layer.length < 2) {
            System.out.println("Arsitektur NN belum terbentuk dengan benar");
            return;
        }

        int outputNum = this.m_layer[this.m_layer.length - 1].neuron.length - 1;

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
            this.m_layer[this.m_layer.length - 1].neuron[i].hitungGalatOutput(targetVal[i]);
        }

        //hitung galat hidden layer
        for (int layerNum = this.m_layer.length - 2; layerNum >= 0; layerNum--) {
            Layer layerSelanjutnya = this.m_layer[layerNum + 1];

            for (int i = 0; i < this.m_layer[layerNum].neuron.length; i++) {
                this.m_layer[layerNum].neuron[i].hitungGalatHidden(layerSelanjutnya);
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

    class Layer {

        public Neuron[] neuron;
    }

    class Connection {

        private double weight;
        private double deltaweight;

        public double getWeight() {
            return weight;
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }

        public double getDeltaweight() {
            return deltaweight;
        }

        public void setDeltaweight(double deltaweight) {
            this.deltaweight = deltaweight;
        }
    }

    class Neuron {

        private Connection[] m_outputWeights;
        private double m_outputVal;
        private int m_myIndex;
        private double m_galat;

        public Neuron() {
        }

        public Neuron(int numOutput, int myIndex) {
            this.m_outputWeights = new Connection[numOutput];
            for (int i = 0; i < numOutput; i++) {
                this.m_outputWeights[i] = new Connection();
                this.m_outputWeights[i].setWeight(this.weightAcak());
            }
            this.m_myIndex = myIndex;
        }

        public void updateWeight(double galat, int i) {

            double deltaWeightDulu = this.m_outputWeights[i].getDeltaweight();

            double deltaWeightBaru =
                    Net.learningrate
                    * this.getOutputVal()
                    * galat
                    + Net.momentum
                    * deltaWeightDulu;

            this.m_outputWeights[i].setDeltaweight(deltaWeightBaru);
            deltaWeightBaru += this.m_outputWeights[i].getWeight();
            this.m_outputWeights[i].setWeight(deltaWeightBaru);


        }

        public void hitungGalatHidden(Layer layerSelanjutnya) {
            double sum = 0.0;

            for (int i = 0; i < layerSelanjutnya.neuron.length - 1; i++) {
                sum += this.m_outputWeights[i].getWeight()
                        * layerSelanjutnya.neuron[i].getGalat();
            }

            this.m_galat = sum * this.transferFunctionTurunan(this.getOutputVal());
        }

        public void hitungGalatOutput(double targetVal) {
            double delta = targetVal - this.getOutputVal();
            this.m_galat = delta * this.transferFunctionTurunan(this.m_outputVal);
        }

        public void feedForward(Layer layerSebelumnya) {
            double sum = 0.0;

            for (int i = 0; i < layerSebelumnya.neuron.length; i++) {
                sum += layerSebelumnya.neuron[i].getOutputVal()
                        * layerSebelumnya.neuron[i].getOutputWeight(this.m_myIndex).getWeight();
            }

            this.m_outputVal = transferFunction(sum);
        }

        private double transferFunction(double x) {
            //sigmoid
            return 1 / (1 + Math.exp(-1 * x));
        }

        private double transferFunctionTurunan(double x) {
            //turunan sigmoid
            return this.transferFunction(x) * (1 - this.transferFunction(x));
        }

        private double weightAcak() {
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
            return this.m_galat;
        }

        public Connection getOutputWeight(int index) {
            return this.m_outputWeights[index];
        }
    }

    public static void main(String[] args) {
        // TODO code application logic here
        Net net = new Net();
        int[] topology = new int[3];
        topology[0] = 2;
        topology[1] = 4;
        topology[2] = 1;
        //System.out.println(topology.length);
        net.setTopology(topology);

        double[][] input = new double[4][2];
        double[][] target = new double[4][1];

        input[0][0] = 0.0;
        input[0][1] = 0.0;
        target[0][0] = 0.0;

        input[1][0] = 0.0;
        input[1][1] = 1.0;
        target[1][0] = 1.0;

        input[2][0] = 1.0;
        input[2][1] = 0.0;
        target[2][0] = 1.0;

        input[3][0] = 1.0;
        input[3][1] = 1.0;
        target[3][0] = 1.0;

        double error = 0.0;
        int epoc = 0;
        do {
            net.sum_global_error = 0.0;

            for (int i = 0; i < 4; i++) {
                net.feedForward(input[i]);
                net.backProp(target[i]);
            }
            error = Math.sqrt(net.sum_global_error / (4 * topology[topology.length - 1]));
            epoc++;
            if(epoc % 100 == 0){
                System.out.println("learning masih proses di epoc " + epoc + ". dengan error " + error);
            }

        } while ( error > 0.01 );
        
        System.out.println("learning berhenti di epoc " + epoc + ". dengan error " + error);
        
        System.out.println("\npercobaan :");
        double[] hasil;
        for (int i = 0; i < 4; i++) {
            net.feedForward(input[i]);
            hasil = net.getHasil();
            System.out.println( input[i][0] + " or "+input[i][1]+" = " + hasil[0]);
        }        
    }
}
