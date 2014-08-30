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
public class Neuron {

    protected int myIndex;
    protected double outputVal;
    protected Connection[] outputWeights;
    protected double galat;

    public Neuron(int myIndex){
        this.myIndex = myIndex;
    }
    
    public Neuron(int numOutput, int myIndex) {
        //inisialisasi koneksi neuron
        this.outputWeights = new Connection[numOutput];
        for (int i = 0; i < numOutput; i++) {
            this.outputWeights[i] = new Connection();
            this.outputWeights[i].setWeight(Neuron.weightAcak());
        }
        
        this.myIndex = myIndex;
    }

    public void updateWeight(Layer layerSebelumnya) {
        
        for(Neuron neuron : layerSebelumnya.getNeurons()){
            Connection weight = neuron.getOutputWeight(myIndex);
            
            double deltaWeightBaru =
                Net.learningrate
                * neuron.getOutputVal()
                * this.galat
                + Net.momentum
                * weight.getDeltaweight();
            
            weight.setDeltaweight(deltaWeightBaru);
            weight.setWeight(weight.getWeight() + deltaWeightBaru);
        }
    }

    public void feedForward(Layer layerSebelumnya) {
        double sum = layerSebelumnya.getSumInput(myIndex);
        this.outputVal = TransferFunction.transfer(sum);
    }

    protected static double weightAcak() {
        //nilai acak [-1 sampai 1]
        return Math.round(Math.random()) * 2 - 1;
    }

    public double getOutputVal() {
        return this.outputVal;
    }

    public void setOutputVal(double outputVal) {
        this.outputVal = outputVal;
    }

    public double getGalat() {
        return this.galat;
    }

    public Connection getOutputWeight(int index) {
        return this.outputWeights[index];
    }
}
