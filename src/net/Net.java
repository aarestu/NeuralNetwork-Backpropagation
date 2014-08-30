/*
 * Copyright 2014 aarestu.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package net;

/**
 *
 * @author aarestu
 */
public class Net {

    public static double learningrate = 0.25;
    public static double momentum = 0.5;
    private Layer[] layers;
    private double myError;
    public double sum_global_error;

    public Net() {
    }

    public void setTopology(int[] topology) {
        int numLayers = topology.length;
        if (numLayers < 3) {
            System.out.println("minimal 3 layer : input, hidden, output");
            return;
        }

        this.layers = new Layer[numLayers];

        // Inisialisasi neuron dan bias untuk input layer
        Neuron[] inputNeuron = new Neuron[topology[0] + 1];
        int numOutput = topology[1];

        for (int neuronNum = 0; neuronNum <= topology[0]; neuronNum++) {
            inputNeuron[neuronNum] = new Neuron(numOutput, neuronNum);
        }

        this.layers[0] = new Layer(inputNeuron);

        // neuron terakhir berperan sebagai bias, isi output = 1
        this.layers[0].getIndexNeuron(topology[0]).setOutputVal(1);

        // Inisialisasi hidden layer
        for (int layerNum = 1; layerNum < numLayers - 1; layerNum++) {

            // Buat neuron dan bias untuk setiap layer
            NeuronHidden[] hiddenNeuron = new NeuronHidden[topology[layerNum] + 1];
            numOutput = topology[layerNum + 1];

            for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
                hiddenNeuron[neuronNum] = new NeuronHidden(numOutput, neuronNum);
            }

            this.layers[layerNum] = new Layer(hiddenNeuron);
            //neuron terakhir berperan sebagai bias, isi output = 1
            this.layers[layerNum].getIndexNeuron(topology[layerNum]).setOutputVal(1);
        }

        // Inisialisasi output layer
        NeuronOutput[] outputNeuron = new NeuronOutput[topology[numLayers - 1] + 1];

        for (int neuronNum = 0; neuronNum <= topology[numLayers - 1]; neuronNum++) {
            outputNeuron[neuronNum] = new NeuronOutput(neuronNum);
        }

        this.layers[numLayers - 1] = new Layer(outputNeuron);
        
    }

    public void backProp(double[] targetVal) {
        if (this.layers == null || this.layers.length < 3) {
            System.out.println("Arsitektur NN belum terbentuk dengan benar");
            return;
        }

        int outputNum = this.layers[this.layers.length - 1].getSizeNeuron() - 1;

        if (targetVal.length != outputNum) {
            System.out.println("banyak target tidak sama dengan banyak neuron di output layer");
            return;
        }


        this.myError = 0.0;
        for (int i = 0; i < outputNum; i++) {

            NeuronOutput no = (NeuronOutput) this.layers[this.layers.length - 1].getIndexNeuron(i);
            no.setTarget(targetVal[i]);

            //hitung error local
            double delta = no.getOutputError();
            this.myError += delta * delta;
            this.sum_global_error += this.myError;

            //hitung galat output layer
            no.hitungGalat();
        }

        //hitung error local
        this.myError = this.myError / outputNum; // Mean Squared Error (MSE)
        this.myError = Math.sqrt(this.myError); // RMS

        //hitung galat hidden layer
        for (int layerNum = this.layers.length - 2; layerNum > 0; layerNum--) {
            Layer hiddenLayer = this.layers[layerNum];
            Layer layerSelanjutnya = this.layers[layerNum + 1];

            for (int i = 0; i < hiddenLayer.getSizeNeuron(); i++) {
                NeuronHidden nh = (NeuronHidden) hiddenLayer.getIndexNeuron(i);
                nh.hitungGalat(layerSelanjutnya);
            }
        }

        //update bobot koneksi atau weight
        for (int layerNum = this.layers.length - 1; layerNum > 0; layerNum--) {
            Layer layer = this.layers[layerNum];
            Layer layerSebelumnya = this.layers[layerNum - 1];

            for (int n = 0; n < layer.getSizeNeuron() - 1; n++) {
                layer.getIndexNeuron(n).updateWeight(layerSebelumnya);
            }
        }
    }

    public void feedForward(double[] inputVal) {
        if (this.layers == null || this.layers.length < 3) {
            System.out.println("Arsitektur NN belum terbentuk dengan benar");
            return;
        }

        if (inputVal.length != this.layers[0].getSizeNeuron() - 1) {
            System.out.println("banyak input tidak sama dengan banyak neuron di input layer");
            return;
        }

        //inisialisasi input layer
        for (int i = 0; i < inputVal.length; i++) {
            this.layers[0].getIndexNeuron(i).setOutputVal(inputVal[i]);
        }

        //feed forwared propagate
        for (int layerNum = 1; layerNum < this.layers.length; layerNum++) {
            Layer layerSebelumnya = this.layers[layerNum - 1];
            for (int n = 0; n < this.layers[layerNum].getSizeNeuron() - 1; n++) {
                this.layers[layerNum].getIndexNeuron(n).feedForward(layerSebelumnya);
            }
        }
    }

    public double getError() {
        return this.myError;
    }

    public double[] getHasil() {
        int outputNum = this.layers[this.layers.length - 1].getSizeNeuron() - 1;
        double[] hasil = new double[outputNum];
        for (int i = 0; i < outputNum; i++) {
            hasil[i] = 0;
            hasil[i] = this.layers[this.layers.length - 1].getIndexNeuron(i).getOutputVal();

        }
        return hasil;
    }
}
