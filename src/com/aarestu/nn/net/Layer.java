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
public class Layer {
    private Neuron[] neuron;
    
    public Layer(Neuron[] neuron){
        this.neuron = neuron;
    }
    
    public int getSizeNeuron(){
        return neuron.length;
    }
    
    public Neuron[] getNeurons(){
        return this.neuron;
    }
    
    public Neuron getIndexNeuron(int i){
          return neuron[i];
    }
    
    public double getSumInput(int indexOutput){
        double sum = 0.0;
        
        for (int i = 0; i < this.getSizeNeuron(); i++) {
            Neuron n = this.getIndexNeuron(i);
            Connection w = n.getOutputWeight(indexOutput);
            
            sum += n.getOutputVal()
                    * w.getWeight();
        }
        
        return sum;
    }
}
