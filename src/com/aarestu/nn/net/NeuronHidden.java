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

package com.aarestu.nn.net;

/**
 *
 * @author aarestu
 */
public class NeuronHidden extends Neuron {
    
    
    public NeuronHidden(int numOutput, int myIndex) {
        super(numOutput, myIndex);
    }
    
    public void hitungGalat(Layer layerSelanjutnya) {
        double sum = 0.0;

        for (int i = 0; i < layerSelanjutnya.getSizeNeuron() - 1; i++) {
            sum += this.outputWeights[i].getWeight()
                    * layerSelanjutnya.getIndexNeuron(i).getGalat();
        }

        this.galat = sum * TransferFunction.transferTurunan(this.getOutputVal());
    }
}
